import re
import logging
import time
from typing import Dict, Any, List, Optional
from core.interfaces.embedding import IEmbeddingService
from core.interfaces.vector_store import IVectorStore
from core.interfaces.llm import ILLMService
from core.interfaces.ocr import IOCRService
from core.interfaces.vision import IVisionService
from pipeline.prompts import (
    render_user_prompt, render_image_prompt, SYSTEM_PROMPT,
    SYSTEM_PROMPT_PREFIX, TEMPLATE_DEFINITION, TEMPLATE_SHORT_FACT,
    TEMPLATE_HOW_TO, TEMPLATE_API_ENDPOINT, TEMPLATE_TROUBLESHOOTING, TEMPLATE_COMPARISON
)
from core.exceptions import TechnicalSupportBotException

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrator for the Retrieval-Augmented Generation (RAG) query loop.
    Extracts text context from pgvector based on query similarities and routes
    grounded prompts to the local LLM.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
        llm_service: ILLMService,
        ocr_service: Optional[IOCRService] = None,
        vision_service: Optional[IVisionService] = None
    ):
        """
        Initialize the RAG Pipeline.

        Args:
            embedding_service: Service to generate vector embeddings.
            vector_store: Target vector database store.
            llm_service: Large Language Model client service.
            ocr_service: Pluggable OCR transcribing service.
            vision_service: Pluggable visual scene analyzer service.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.ocr_service = ocr_service
        self.vision_service = vision_service

    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Run the end-to-end RAG query flow.

        Args:
            user_query: The technical support question.
            top_k: Number of similarity chunks to retrieve from vector database.

        Returns:
            A structured dictionary containing:
                - query: The original user question.
                - answer: Synthesized grounded response.
                - sources: List of source document metadata records.
                - metrics: Dictionary of timing metrics (seconds).
        """
        logger.info("Request received")
        start_pipeline = time.time()

        try:
            # 1. Generate query embedding
            logger.info("RAG Pipeline: Generating query vector...")
            start_emb = time.time()
            query_vector = self.embedding_service.embed_query(user_query)
            emb_duration = time.time() - start_emb
            logger.info(f"RAG Pipeline: Embedding generated in {emb_duration:.3f}s")

            logger.info(f"RAG Pipeline: Executing pgvector similarity search (top_k={top_k})...")
            start_ret = time.time()
            retrieved_candidates = self.vector_store.similarity_search(query_vector, limit=top_k * 3)
            retrieved_results = self._rerank_results(retrieved_candidates, user_query, top_k)
            ret_duration = time.time() - start_ret
            
            # Filter low-similarity results (threshold = 0.25)
            retrieved_results = [res for res in retrieved_results if res.similarity_score >= 0.25]
            
            # Filter for OAuth-specific queries to prevent distractor endpoints
            query_lower = user_query.lower()
            is_oauth_query = any(k in query_lower for k in ["oauth", "ouath", "token", "login", "auth", "credential"])
            if is_oauth_query:
                oauth_results = []
                for res in retrieved_results:
                    meta = res.chunk.metadata or {}
                    category = (meta.get("category") or "").lower()
                    chunk_text = res.chunk.content.lower()
                    if category == "authentication" or any(k in chunk_text for k in ["oauth", "ouath", "token", "client_credentials"]):
                        oauth_results.append(res)
                retrieved_results = oauth_results
                
            logger.info(f"RAG Pipeline: Retrieved and re-ranked {len(retrieved_results)} chunks in {ret_duration:.3f}s")

            # Temporary debug logging for retrieved chunks
            logger.info("=== TEMPORARY DEBUG LOGGING: RETRIEVED CHUNKS ===")
            for idx, res in enumerate(retrieved_results):
                chunk = res.chunk
                logger.info(f"Chunk {idx+1}:")
                logger.info(f"  - Chunk ID: {chunk.id}")
                logger.info(f"  - Source Document: {chunk.metadata.get('filename') if chunk.metadata else 'Unknown'}")
                logger.info(f"  - Similarity Score: {res.similarity_score}")
                logger.info(f"  - Content: {chunk.content}")
            logger.info("==================================================")

            # 3. Construct prompt
            logger.info("RAG Pipeline: Formatting prompt with retrieved chunks...")
            start_prompt = time.time()
            chunks = [res.chunk for res in retrieved_results]
            user_prompt = render_user_prompt(user_query, chunks)
            prompt_duration = time.time() - start_prompt

            # Diagnostic Logging
            context_len = sum(len(c.content) for c in chunks)
            estimated_tokens = len(user_prompt) // 4
            logger.info(f"[DIAGNOSTIC LOG] Prompt length: {len(user_prompt)} characters")
            logger.info(f"[DIAGNOSTIC LOG] Context length: {context_len} characters")
            logger.info(f"[DIAGNOSTIC LOG] Number of retrieved chunks: {len(chunks)}")
            logger.info(f"[DIAGNOSTIC LOG] Prompt preview (first 300 chars): {user_prompt[:300]}...")

            # Apply dynamic guidance and template pruning to reduce context size and CPU prefill time
            query_lower = user_query.lower()
            is_comparison = any(k in query_lower for k in ["vs", "compare", "difference", "comparison", "different"])
            is_troubleshooting = any(k in query_lower for k in ["error", "fail", "401", "403", "404", "500", "invalid", "trouble", "why", "user cannot", "cannot access", "not work"])
            is_howto = any(k in query_lower for k in ["how to", "how do i", "how can i", "steps to", "procedure"])
            is_api_endpoint = any(k in query_lower for k in ["endpoint", "api", "url", "post", "get", "put", "delete", "request", "payload", "method", "headers"])
            is_short_fact = any(k in query_lower for k in ["what is the endpoint", "where is", "which api", "oauth endpoint", "token endpoint"])

            # Dynamically select only the classified response template. This reduces prompt size by >50%.
            if is_comparison:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_COMPARISON
                max_tokens_override = 300
            elif is_troubleshooting:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_TROUBLESHOOTING
                max_tokens_override = 250
            elif is_howto:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_HOW_TO
                max_tokens_override = 250
            elif is_short_fact:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_SHORT_FACT
                max_tokens_override = 60
            elif is_api_endpoint:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_API_ENDPOINT
                max_tokens_override = 300
            else:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_DEFINITION
                max_tokens_override = 120

            # 4. Generate response from LLM
            logger.info("RAG Pipeline: Sending prompt to LLM...")
            start_llm = time.time()
            llm_response = self.llm_service.generate_response(
                user_prompt, 
                system_prompt=custom_sys_prompt, 
                max_tokens=max_tokens_override
            )
            llm_duration = time.time() - start_llm
            logger.info(f"RAG Pipeline: LLM generated response in {llm_duration:.3f}s")

            # Compile source citations (deduplicating sources for clean presentation)
            sources = []
            seen_source_keys = set()
            for res in retrieved_results:
                chunk = res.chunk
                filename = chunk.metadata.get("filename", "unknown")
                page = chunk.metadata.get("page_number")
                api_path = chunk.metadata.get("api_path")
                method = chunk.metadata.get("http_method")
                section = chunk.metadata.get("section")
                
                # Create a uniqueness key to avoid duplicate citations
                source_key = (filename, page, api_path, method, section)
                if source_key not in seen_source_keys:
                    seen_source_keys.add(source_key)
                    sources.append({
                        "document_id": str(chunk.document_id),
                        "filename": filename,
                        "similarity_score": round(res.similarity_score, 4),
                        "api_path": api_path,
                        "http_method": method,
                        "page_number": page,
                        "section": section
                    })

            # Response formatting
            start_format = time.time()
            formatted_answer = llm_response.strip().replace("\\n", "\n").replace('\\"', '"').replace('\\`', '`')
            format_duration = time.time() - start_format

            pipeline_duration = time.time() - start_pipeline
            logger.info("RAG Pipeline: Query processing completed successfully.")

            # Print exact requested timings formatting to console/stdout
            print("\nRequest received\n")
            print(f"Embedding generation:\n{int(emb_duration * 1000)} ms\n")
            print(f"Vector search:\n{int(ret_duration * 1000)} ms\n")
            print(f"Prompt construction:\n{int(prompt_duration * 1000)} ms\n")
            print(f"Prompt size:\n{len(user_prompt)} characters\n")
            print(f"Estimated prompt tokens:\n{estimated_tokens}\n")
            print(f"LLM generation:\n{llm_duration:.2f} seconds\n")
            print(f"Response formatting:\n{int(format_duration * 1000)} ms\n")
            print(f"Total execution:\n{pipeline_duration:.2f} seconds\n")

            return {
                "query": user_query,
                "answer": formatted_answer,
                "sources": sources,
                "metrics": {
                    "embedding_time_seconds": round(emb_duration, 3),
                    "retrieval_time_seconds": round(ret_duration, 3),
                    "llm_time_seconds": round(llm_duration, 3),
                    "total_time_seconds": round(pipeline_duration, 3),
                    "chunks_retrieved": len(retrieved_results)
                }
            }

        except TechnicalSupportBotException as tbe:
            logger.error(f"Application exception in RAG Pipeline: {tbe.message}")
            raise tbe
        except Exception as e:
            err_msg = f"Unexpected failure in RAG Pipeline: {e}"
            logger.error(err_msg, exc_info=True)
            raise TechnicalSupportBotException(err_msg)

    def query_image(self, image_bytes: bytes, user_prompt: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Run the end-to-end multimodal image RAG query flow.
        Transcribes logs via OCR, classifies environment via Vision, matches
        contexts in pgvector, and generates diagnostics via Ollama.

        Args:
            image_bytes: Binary screenshot payload.
            user_prompt: User's optional guiding question.
            top_k: Chunks to pull from vector store.

        Returns:
            Structured diagnostics payload.
        """
        logger.info(f"RAG Pipeline: Image query received. Prompt: '{user_prompt}'")
        start_pipeline = time.time()

        if not self.ocr_service or not self.vision_service:
            raise TechnicalSupportBotException("OCR or Vision service is not configured on this pipeline.")

        try:
            # 1. OCR text extraction
            logger.info("RAG Pipeline: Extracting text via OCR...")
            start_ocr = time.time()
            ocr_result = self.ocr_service.extract_text(image_bytes)
            ocr_duration = time.time() - start_ocr
            logger.info(f"RAG Pipeline: OCR extracted text in {ocr_duration:.3f}s")

            # 2. Vision layout analysis
            logger.info("RAG Pipeline: Analyzing image layout via Vision...")
            start_vis = time.time()
            vision_result = self.vision_service.analyze_image(image_bytes, prompt=user_prompt)
            vis_duration = time.time() - start_vis
            logger.info(f"RAG Pipeline: Vision analysis completed in {vis_duration:.3f}s")

            # 3. Combine query elements and generate embedding
            combined_query_parts = []
            if user_prompt:
                combined_query_parts.append(f"User Query: {user_prompt}")
            if vision_result.description:
                combined_query_parts.append(f"Screen Description: {vision_result.description}")
            if vision_result.detected_errors:
                combined_query_parts.append(f"Detected Exception: {', '.join(vision_result.detected_errors)}")
            if ocr_result.extracted_text:
                combined_query_parts.append(f"Error Code logs:\n{ocr_result.extracted_text}")
                
            combined_query = "\n\n".join(combined_query_parts)
            
            logger.info("RAG Pipeline: Generating combined query vector...")
            start_emb = time.time()
            query_vector = self.embedding_service.embed_query(combined_query)
            emb_duration = time.time() - start_emb
            logger.info(f"RAG Pipeline: Embedding generated in {emb_duration:.3f}s")

            logger.info(f"RAG Pipeline: Executing pgvector similarity search (top_k={top_k})...")
            start_ret = time.time()
            retrieved_candidates = self.vector_store.similarity_search(query_vector, limit=top_k * 3)
            retrieved_results = self._rerank_results(retrieved_candidates, user_prompt or "", top_k)
            ret_duration = time.time() - start_ret
            
            # Filter low-similarity results (threshold = 0.25)
            retrieved_results = [res for res in retrieved_results if res.similarity_score >= 0.25]
            
            # Filter for OAuth-specific queries to prevent distractor endpoints
            combined_query_lower = combined_query.lower()
            is_oauth_query = any(k in combined_query_lower for k in ["oauth", "ouath", "token", "login", "auth", "credential"])
            if is_oauth_query:
                oauth_results = []
                for res in retrieved_results:
                    meta = res.chunk.metadata or {}
                    category = (meta.get("category") or "").lower()
                    chunk_text = res.chunk.content.lower()
                    if category == "authentication" or any(k in chunk_text for k in ["oauth", "ouath", "token", "client_credentials"]):
                        oauth_results.append(res)
                retrieved_results = oauth_results
                
            logger.info(f"RAG Pipeline: Retrieved and re-ranked {len(retrieved_results)} chunks in {ret_duration:.3f}s")

            # Temporary debug logging for retrieved chunks
            logger.info("=== TEMPORARY DEBUG LOGGING: RETRIEVED CHUNKS ===")
            for idx, res in enumerate(retrieved_results):
                chunk = res.chunk
                logger.info(f"Chunk {idx+1}:")
                logger.info(f"  - Chunk ID: {chunk.id}")
                logger.info(f"  - Source Document: {chunk.metadata.get('filename') if chunk.metadata else 'Unknown'}")
                logger.info(f"  - Similarity Score: {res.similarity_score}")
                logger.info(f"  - Content: {chunk.content}")
            logger.info("==================================================")

            # 5. Construct prompt
            logger.info("RAG Pipeline: Formatting prompt with retrieved chunks...")
            chunks = [res.chunk for res in retrieved_results]
            image_prompt = render_image_prompt(
                user_question=user_prompt,
                ocr_text=ocr_result.extracted_text,
                vision_description=vision_result.description,
                chunks=chunks
            )

            # Diagnostic Logging (Task 3 & 7 requirement)
            context_len = sum(len(c.content) for c in chunks)
            logger.info(f"[DIAGNOSTIC LOG] Image Prompt length: {len(image_prompt)} characters")
            logger.info(f"[DIAGNOSTIC LOG] Context length: {context_len} characters")
            logger.info(f"[DIAGNOSTIC LOG] Number of retrieved chunks: {len(chunks)}")
            logger.info(f"[DIAGNOSTIC LOG] Prompt preview (first 300 chars): {image_prompt[:300]}...")

            # Apply dynamic guidance and template pruning to reduce context size and CPU prefill time
            query_lower = (user_prompt or "").lower()
            is_comparison = any(k in query_lower for k in ["vs", "compare", "difference", "comparison", "different"])
            is_howto = any(k in query_lower for k in ["how to", "how do i", "how can i", "steps to", "procedure"])
            is_api_endpoint = any(k in query_lower for k in ["endpoint", "api", "url", "post", "get", "put", "delete", "request", "payload", "method", "headers"])

            # Dynamically select only the classified response template.
            if is_comparison:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_COMPARISON
                max_tokens_override = 300
            elif is_howto:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_HOW_TO
                max_tokens_override = 250
            elif is_api_endpoint:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_API_ENDPOINT
                max_tokens_override = 300
            else:
                custom_sys_prompt = SYSTEM_PROMPT_PREFIX + "1. " + TEMPLATE_TROUBLESHOOTING
                max_tokens_override = 250

            # 6. Generate response from LLM
            logger.info("RAG Pipeline: Sending prompt to LLM...")
            start_llm = time.time()
            llm_response = self.llm_service.generate_response(
                image_prompt, 
                system_prompt=custom_sys_prompt, 
                max_tokens=max_tokens_override
            )
            llm_duration = time.time() - start_llm
            logger.info(f"RAG Pipeline: LLM generated response in {llm_duration:.3f}s")

            # Compile source citations (deduplicating sources)
            sources = []
            seen_source_keys = set()
            for res in retrieved_results:
                chunk = res.chunk
                filename = chunk.metadata.get("filename", "unknown")
                page = chunk.metadata.get("page_number")
                api_path = chunk.metadata.get("api_path")
                method = chunk.metadata.get("http_method")
                section = chunk.metadata.get("section")
                
                source_key = (filename, page, api_path, method, section)
                if source_key not in seen_source_keys:
                    seen_source_keys.add(source_key)
                    sources.append({
                        "document_id": str(chunk.document_id),
                        "filename": filename,
                        "similarity_score": round(res.similarity_score, 4),
                        "api_path": api_path,
                        "http_method": method,
                        "page_number": page,
                        "section": section
                    })

            pipeline_duration = time.time() - start_pipeline
            logger.info("RAG Pipeline: Image query processing completed successfully.")

            return {
                "query": user_prompt,
                "extracted_text": ocr_result.extracted_text,
                "vision_description": vision_result.description,
                "answer": llm_response.strip().replace("\\n", "\n").replace('\\"', '"').replace('\\`', '`'),
                "sources": sources,
                "metrics": {
                    "ocr_time_seconds": round(ocr_duration, 3),
                    "vision_time_seconds": round(vis_duration, 3),
                    "embedding_time_seconds": round(emb_duration, 3),
                    "retrieval_time_seconds": round(ret_duration, 3),
                    "llm_time_seconds": round(llm_duration, 3),
                    "total_time_seconds": round(pipeline_duration, 3),
                    "chunks_retrieved": len(retrieved_results)
                }
            }

        except TechnicalSupportBotException as tbe:
            logger.error(f"Application exception in Image RAG Pipeline: {tbe.message}")
            raise tbe
        except Exception as e:
            err_msg = f"Unexpected failure in Image RAG Pipeline: {e}"
            logger.error(err_msg, exc_info=True)
            raise TechnicalSupportBotException(err_msg)

    def _rerank_results(self, results: List[Any], user_query: str, top_k: int) -> List[Any]:
        """
        Re-ranks retrieved vector chunks by boosting similarity scores for chunks
        where metadata fields (endpoint_name, category, http_method) match the user query.
        """
        if not results:
            return []
            
        query_lower = user_query.lower()
        
        for res in results:
            chunk = res.chunk
            meta = chunk.metadata or {}
            
            # Retrieve metadata elements
            endpoint_name = (meta.get("endpoint_name") or "").lower()
            category = (meta.get("category") or "").lower()
            http_method = (meta.get("http_method") or "").lower()
            url_path = (meta.get("url") or meta.get("api_path") or "").lower()
            
            boost = 0.0
            
            # Boost if query explicitly targets the category name
            if category and category in query_lower:
                boost += 0.15
            elif category == "authentication" and any(k in query_lower for k in ["login", "token", "auth"]):
                boost += 0.15
            elif category == "user management" and any(k in query_lower for k in ["user", "role", "permission"]):
                boost += 0.15
            elif category == "access control" and any(k in query_lower for k in ["access", "reader", "door"]):
                boost += 0.15
            elif category == "card assignment" and any(k in query_lower for k in ["card", "credential", "badge"]):
                boost += 0.15
                
            # Boost if query matches endpoint name or path variables
            if endpoint_name and endpoint_name in query_lower:
                boost += 0.20
            # Also boost if the query matches the url/path endpoint exactly
            if url_path and url_path in query_lower:
                boost += 0.25
                
            # Boost if query specifies the exact HTTP method used
            if http_method and re.search(r'\b' + re.escape(http_method) + r'\b', query_lower):
                boost += 0.10
                
            res.similarity_score += boost
            
        # Re-sort results descending by the boosted score
        reranked = sorted(results, key=lambda x: x.similarity_score, reverse=True)
        return reranked[:top_k]
