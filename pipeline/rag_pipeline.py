import re
import logging
import time
import json
from typing import Dict, Any, List, Optional
from core.interfaces.embedding import IEmbeddingService
from core.interfaces.vector_store import IVectorStore
from core.interfaces.llm import ILLMService
from core.interfaces.ocr import IOCRService
from core.interfaces.vision import IVisionService
from pipeline.prompts import (
    render_user_prompt, render_image_prompt, SYSTEM_PROMPT, format_fact_sheet,
    is_operation_match
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
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.ocr_service = ocr_service
        self.vision_service = vision_service

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """
        Unifies classification of the user's query.
        Returns:
            {
                "target_operation": str,  # 'oauth_token', 'create_user', etc. or 'None'
                "query_type": str,        # 'how_to', 'api_endpoint', etc.
                "category": str           # 'authentication', 'user management', etc.
            }
        """
        query_lower = query.lower()
        target_operation = "None"
        category = "general"
        
        # OAuth Token
        if any(k in query_lower for k in ["oauth", "access token", "oauth token", "auth token", "clientid", "clientsecret"]):
            target_operation = "oauth_token"
            category = "authentication"
            
        # Access Points (Specific)
        elif "access point" in query_lower or "access points" in query_lower:
            category = "access control"
            if "grant" in query_lower or "add" in query_lower or "assign" in query_lower:
                target_operation = "grant_access_point_permissions"
            elif "revoke" in query_lower or "remove" in query_lower:
                target_operation = "revoke_access_point_permissions"
            else:
                target_operation = "get_access_points"

        # User permissions (Specific)
        elif "user permissions" in query_lower or "get user permissions" in query_lower or "retrieve user permissions" in query_lower:
            target_operation = "get_user_permissions"
            category = "access control"
            
        # User Management
        elif "user" in query_lower or "users" in query_lower or "employee" in query_lower:
            category = "user management"
            if any(k in query_lower for k in ["deactivate", "disable", "suspend"]):
                target_operation = "deactivate_user"
            elif any(k in query_lower for k in ["activate", "enable", "re-enable", "re-activate"]):
                target_operation = "activate_user"
            elif any(k in query_lower for k in ["delete", "remove", "permanently"]):
                target_operation = "delete_user"
            elif any(k in query_lower for k in ["create", "add", "onboard", "register"]):
                target_operation = "create_user"
            elif any(k in query_lower for k in ["update", "modify", "edit", "patch"]):
                target_operation = "update_user"
            elif any(k in query_lower for k in ["permission", "permissions"]):
                if "grant" in query_lower or "add" in query_lower or "assign" in query_lower:
                    target_operation = "grant_access_point_permissions"
                    category = "access control"
                elif "revoke" in query_lower or "remove" in query_lower:
                    target_operation = "revoke_access_point_permissions"
                    category = "access control"
                else:
                    target_operation = "get_user_permissions"
                    category = "access control"
            elif any(k in query_lower for k in ["list", "fetch", "get all", "retrieve all"]):
                target_operation = "fetch_all_organisation_users"
                
        # General Permissions / Access Control (non-user keyword context)
        elif "permission" in query_lower or "permissions" in query_lower or "door" in query_lower or "reader" in query_lower:
            category = "access control"
            if "grant" in query_lower or "add" in query_lower or "assign" in query_lower:
                target_operation = "grant_access_point_permissions"
            elif "revoke" in query_lower or "remove" in query_lower or "delete" in query_lower:
                target_operation = "revoke_access_point_permissions"
            elif "get" in query_lower or "view" in query_lower or "show" in query_lower:
                target_operation = "get_user_permissions"
            else:
                target_operation = "get_access_points"
                
        # VMS Meeting
        elif "meeting" in query_lower or "meetings" in query_lower:
            category = "meeting"
            if any(k in query_lower for k in ["create", "add", "schedule", "book"]):
                target_operation = "create_meeting"
            elif any(k in query_lower for k in ["update", "modify", "edit", "patch"]):
                target_operation = "update_meeting"
            elif any(k in query_lower for k in ["delete", "cancel", "remove"]):
                target_operation = "delete_meeting"
                
        # Sites & Roles
        elif "site" in query_lower or "sites" in query_lower:
            target_operation = "get_sites"
            category = "organisation"
        elif "role" in query_lower or "roles" in query_lower:
            target_operation = "get_roles"
            category = "organisation"
            
        # Query Type Intent
        query_type = "definition"
        is_comparison = any(k in query_lower for k in ["vs", "compare", "difference", "comparison", "different"])
        is_troubleshooting = any(k in query_lower for k in ["error", "fail", "401", "403", "404", "500", "invalid", "trouble", "why", "user cannot", "cannot access", "not work"])
        is_howto = any(k in query_lower for k in ["how to", "how do i", "how can i", "steps to", "procedure"])
        is_api_endpoint = any(k in query_lower for k in ["endpoint", "api", "url", "post", "get", "put", "delete", "request", "payload", "method", "headers"])
        is_short_fact = any(k in query_lower for k in ["what is the endpoint", "where is", "which api", "oauth endpoint", "token endpoint"])
        
        if is_comparison:
            query_type = "comparison"
        elif is_troubleshooting:
            query_type = "troubleshooting"
        elif is_howto:
            query_type = "how_to"
        elif is_short_fact:
            query_type = "short_fact"
        elif is_api_endpoint:
            query_type = "api_endpoint"
            
        return {
            "target_operation": target_operation,
            "query_type": query_type,
            "category": category
        }

    def _validate_response(self, response_text: str, target_operation: str, fact_sheet: str) -> bool:
        """
        Validates LLM response text against target operation rules.
        """
        response_lower = response_text.lower()
        
        # 1. Obvious recursive JSON corruption check
        if response_text.count("{") > 10 or response_text.count("}") > 10:
            return False
            
        # 2. Check for conflicting operations
        if target_operation == "deactivate_user":
            if "delete" in response_lower:
                return False
            if "deactivateuser" in response_lower and "false" in response_lower:
                return False
        elif target_operation == "activate_user":
            if "delete" in response_lower:
                return False
            if "deactivateuser" in response_lower and "true" in response_lower:
                return False
        elif target_operation == "delete_user":
            if "update" in response_lower or "deactivate" in response_lower:
                return False
            if "userids" not in response_lower:
                return False
        elif target_operation == "grant_access_point_permissions":
            if "permissionstoremove" in response_lower and not '"permissionstoremove": []' in response_text.replace(" ", "").replace("\n", ""):
                return False
            if "userid" in response_lower or "homesiteid" in response_lower:
                return False
        elif target_operation == "revoke_access_point_permissions":
            if "permissionstoadd" in response_lower and not '"permissionstoadd": []' in response_text.replace(" ", "").replace("\n", ""):
                return False
            if "userid" in response_lower or "homesiteid" in response_lower:
                return False
        elif target_operation == "update_user":
            if "users" not in response_lower:
                return False
        elif target_operation == "oauth_token":
            if "browser" in response_lower or "submit" in response_lower:
                return False
            # Ensure "No Auth" is mentioned if configuration steps are detailed
            if "steps" in response_lower or "authorization" in response_lower:
                if "no auth" not in response_lower:
                    return False
                if any(phrase in response_lower for phrase in ["bearer token under authorization", "bearer token as the authorization", "bearer token authorization", "configure the authorization header with the bearer"]):
                    return False
                
        # 3. Check for presence of key URL/method components
        method_match = re.search(r"- \*\*HTTP Method\*\*:\s*([A-Z]+)", fact_sheet)
        if method_match:
            method = method_match.group(1)
            if method.lower() not in response_lower:
                return False
                
        endpoint_match = re.search(r"- \*\*Endpoint\*\*:\s*(\S+)", fact_sheet)
        if endpoint_match:
            url = endpoint_match.group(1)
            path_segments = [s for s in url.split("/") if s and not s.startswith("http")]
            if path_segments:
                last_segment = path_segments[-1].split("?")[0].lower()
                if last_segment not in response_lower:
                    return False
                    
        # 4. JSON request body syntax check
        json_blocks = re.findall(r"```json\s*\n(.*?)\n```", response_text, re.DOTALL)
        for block in json_blocks:
            clean_block = re.sub(r'<.*?>', 'placeholder_value', block)
            try:
                json.loads(clean_block)
            except json.JSONDecodeError:
                return False
                
        # 5. JSON request body completeness verification (verifies specific target specs body fields)
        if "steps" in response_lower or "body" in response_lower:
            if target_operation == "create_user":
                if not all(f in response_lower for f in ["name", "phone", "roles", "adminofsites", "devicelock", "accessdata", "homesiteid", "employeecode"]):
                    return False
            elif target_operation == "update_user":
                if not all(f in response_lower for f in ["users", "id", "name", "employeecode", "homesiteid", "roles"]):
                    return False
            elif target_operation == "activate_user":
                if not all(f in response_lower for f in ["users", "id", "deactivateuser", "employeecode", "name", "homesiteid", "roles", "accessdata"]):
                    return False
            elif target_operation == "deactivate_user":
                if not all(f in response_lower for f in ["users", "id", "deactivateuser"]):
                    return False
            elif target_operation == "delete_user":
                if "userids" not in response_lower:
                    return False
            elif target_operation == "grant_access_point_permissions":
                if not all(f in response_lower for f in ["permissionstoadd", "permissionstoremove", "pendingpermissionstoremove"]):
                    return False
            elif target_operation == "revoke_access_point_permissions":
                if not all(f in response_lower for f in ["permissionstoadd", "permissionstoremove", "pendingpermissionstoremove"]):
                    return False
            elif target_operation == "get_user_permissions":
                if "sites" not in response_lower:
                    return False
                    
        return True

    def _generate_fallback_response(self, target_operation: str, fact_sheet: str) -> str:
        """
        Generates a deterministic fallback response directly from the verified Fact Sheet.
        """
        method = "POST"
        method_match = re.search(r"- \*\*HTTP Method\*\*:\s*([A-Z]+)", fact_sheet)
        if method_match:
            method = method_match.group(1)
            
        url = "https://saams.api.spintly.com"
        endpoint_match = re.search(r"- \*\*Endpoint\*\*:\s*(\S+)", fact_sheet)
        if endpoint_match:
            url = endpoint_match.group(1)
            
        headers = []
        headers_section = re.search(r"- \*\*Required Headers\*\*:\n(.*?)(?=\n- |$)", fact_sheet, re.DOTALL)
        if headers_section:
            headers = [h.strip().replace("- ", "") for h in headers_section.group(1).split("\n") if h.strip()]
            
        auth = "None"
        auth_match = re.search(r"- \*\*Authorization\*\*:\s*(.*?)\n", fact_sheet)
        if auth_match:
            auth = auth_match.group(1).strip()
            
        body = ""
        body_match = re.search(r"```json\s*\n(.*?)\n```", fact_sheet, re.DOTALL)
        if body_match:
            body = body_match.group(1).strip()
            
        output = []
        output.append(f"Here is the verified API procedure for `{target_operation}`:")
        output.append("\n## Step-by-Step Instructions\n")
        
        step_idx = 1
        output.append(f"{step_idx}. Open Postman and create a new request.")
        step_idx += 1
        output.append(f"{step_idx}. Set the HTTP method to **{method}**.")
        step_idx += 1
        output.append(f"{step_idx}. Enter the endpoint: `{url}`")
        
        if auth and auth != "None":
            step_idx += 1
            output.append(f"{step_idx}. Under the **Authorization** tab, configure the type as `{auth}`.")
            
        if headers:
            step_idx += 1
            h_lines = [f"  - `{h}`" for h in headers]
            output.append(f"{step_idx}. Under the **Headers** tab, add:\n" + "\n".join(h_lines))
            
        if body:
            step_idx += 1
            output.append(f"{step_idx}. Under the **Body** tab, select **raw** and set the format to **JSON**.")
            
        step_idx += 1
        output.append(f"{step_idx}. Click **Send**.")
        
        if body:
            output.append("\n## Example Request\n```json\n" + body + "\n```")
            
        output.append("\n## Example Response\n")
        output.append("[Refer to the retrieved Spintly documentation for example responses]")
        
        return "\n".join(output)

    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Run the end-to-end RAG query flow.
        """
        logger.info("Request received")
        start_pipeline = time.time()

        try:
            # 1. Unified classification of user query
            classification = self._classify_query(user_query)
            target_operation = classification["target_operation"]
            query_type = classification["query_type"]
            category = classification["category"]
            
            logger.info(f"RAG Pipeline Classify: Target Op={target_operation}, Query Type={query_type}, Category={category}")

            # 2. Generate query embedding
            logger.info("RAG Pipeline: Generating query vector...")
            start_emb = time.time()
            query_vector = self.embedding_service.embed_query(user_query)
            emb_duration = time.time() - start_emb
            
            # 3. Retrieve database candidates
            logger.info(f"RAG Pipeline: Executing pgvector similarity search (top_k={top_k})...")
            start_ret = time.time()
            retrieved_candidates = self.vector_store.similarity_search(query_vector, limit=max(30, top_k * 6))
            retrieved_results = self._rerank_results(retrieved_candidates, user_query, top_k, target_operation, category)
            ret_duration = time.time() - start_ret
            
            # Filter low-similarity results (threshold = 0.25)
            retrieved_results = [res for res in retrieved_results if res.similarity_score >= 0.25]
            
            # Strict Context Isolation
            filtered_results = []
            if target_operation != "None":
                for res in retrieved_results:
                    meta = res.chunk.metadata or {}
                    endpoint_name = meta.get("endpoint_name") or meta.get("operation_id") or meta.get("request_name") or ""
                    
                    is_exact_match = is_operation_match(endpoint_name, target_operation)
                    
                    is_content_match = False
                    content_lower = res.chunk.content.lower()
                    if target_operation == "oauth_token":
                        is_content_match = any(k in content_lower for k in ["oauth/token", "oauth verification", "client_credentials"])
                    elif target_operation == "deactivate_user":
                        is_content_match = "deactivate" in content_lower and "user" in content_lower
                    elif target_operation == "activate_user":
                        is_content_match = "activate" in content_lower and "user" in content_lower and not "deactivate" in content_lower
                    elif target_operation == "delete_user":
                        is_content_match = "delete user" in content_lower or "delete_user" in content_lower
                        
                    if is_exact_match or is_content_match:
                        filtered_results.append(res)
                        
                if filtered_results:
                    retrieved_results = filtered_results

            logger.info(f"RAG Pipeline: Retrieved and isolated {len(retrieved_results)} chunks.")

            # 4. Construct prompt
            logger.info("RAG Pipeline: Formatting prompt with retrieved chunks...")
            start_prompt = time.time()
            chunks = [res.chunk for res in retrieved_results]
            user_prompt = render_user_prompt(user_query, chunks, target_operation=target_operation, query_type=query_type)
            prompt_duration = time.time() - start_prompt

            context_len = sum(len(c.content) for c in chunks)
            estimated_tokens = len(user_prompt) // 4

            # Token override rules (No Artificial Brevity for HOW-TO questions)
            max_tokens_override = 1000 if query_type in ["how_to", "api_endpoint"] else 800

            # 5. Generate response from LLM
            logger.info("RAG Pipeline: Sending prompt to LLM...")
            start_llm = time.time()
            llm_response = self.llm_service.generate_response(
                user_prompt, 
                system_prompt=SYSTEM_PROMPT, 
                max_tokens=max_tokens_override
            )
            llm_duration = time.time() - start_llm
            logger.info(f"RAG Pipeline: LLM generated response in {llm_duration:.3f}s")

            # Clean and format answer
            start_format = time.time()
            formatted_answer = llm_response.strip().replace("\\n", "\n").replace('\\"', '"').replace('\\`', '`')
            format_duration = time.time() - start_format

            # 6. Response Validation & Fallback Guardrail
            if target_operation != "None" and query_type == "how_to":
                fact_sheet = format_fact_sheet(chunks, target_operation)
                is_valid = self._validate_response(formatted_answer, target_operation, fact_sheet)
                if not is_valid:
                    logger.warning(f"RAG Pipeline: LLM response failed validation for operation '{target_operation}'. Triggering safe fallback.")
                    formatted_answer = self._generate_fallback_response(target_operation, fact_sheet)

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
            
            # 2. Vision layout analysis
            logger.info("RAG Pipeline: Analyzing image layout via Vision...")
            start_vis = time.time()
            vision_result = self.vision_service.analyze_image(image_bytes, prompt=user_prompt)
            vis_duration = time.time() - start_vis
            
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
            
            # Unified classification
            classification = self._classify_query(user_prompt or combined_query)
            target_operation = classification["target_operation"]
            query_type = classification["query_type"]
            category = classification["category"]

            logger.info("RAG Pipeline: Generating combined query vector...")
            start_emb = time.time()
            query_vector = self.embedding_service.embed_query(combined_query)
            emb_duration = time.time() - start_emb
            
            # Retrieve database candidates
            logger.info(f"RAG Pipeline: Executing pgvector similarity search (top_k={top_k})...")
            start_ret = time.time()
            retrieved_candidates = self.vector_store.similarity_search(query_vector, limit=max(30, top_k * 6))
            retrieved_results = self._rerank_results(retrieved_candidates, user_prompt or "", top_k, target_operation, category)
            ret_duration = time.time() - start_ret
            
            # Filter low-similarity results (threshold = 0.25)
            retrieved_results = [res for res in retrieved_results if res.similarity_score >= 0.25]
            
            # Strict Context Isolation
            filtered_results = []
            if target_operation != "None":
                for res in retrieved_results:
                    meta = res.chunk.metadata or {}
                    endpoint_name = meta.get("endpoint_name") or meta.get("operation_id") or meta.get("request_name") or ""
                    is_exact_match = is_operation_match(endpoint_name, target_operation)
                    
                    is_content_match = False
                    content_lower = res.chunk.content.lower()
                    if target_operation == "oauth_token":
                        is_content_match = any(k in content_lower for k in ["oauth/token", "oauth verification", "client_credentials"])
                    elif target_operation == "deactivate_user":
                        is_content_match = "deactivate" in content_lower and "user" in content_lower
                    elif target_operation == "activate_user":
                        is_content_match = "activate" in content_lower and "user" in content_lower and not "deactivate" in content_lower
                    elif target_operation == "delete_user":
                        is_content_match = "delete user" in content_lower or "delete_user" in content_lower
                        
                    if is_exact_match or is_content_match:
                        filtered_results.append(res)
                if filtered_results:
                    retrieved_results = filtered_results

            # 4. Construct prompt
            logger.info("RAG Pipeline: Formatting prompt with retrieved chunks...")
            chunks = [res.chunk for res in retrieved_results]
            image_prompt = render_image_prompt(
                user_question=user_prompt,
                ocr_text=ocr_result.extracted_text,
                vision_description=vision_result.description,
                chunks=chunks,
                target_operation=target_operation,
                query_type=query_type
            )

            context_len = sum(len(c.content) for c in chunks)
            max_tokens_override = 1000 if query_type in ["how_to", "api_endpoint"] else 800

            # 5. Generate response from LLM
            logger.info("RAG Pipeline: Sending prompt to LLM...")
            start_llm = time.time()
            llm_response = self.llm_service.generate_response(
                image_prompt, 
                system_prompt=SYSTEM_PROMPT, 
                max_tokens=max_tokens_override
            )
            llm_duration = time.time() - start_llm

            formatted_answer = llm_response.strip().replace("\\n", "\n").replace('\\"', '"').replace('\\`', '`')

            # 6. Response Validation & Fallback Guardrail
            if target_operation != "None" and query_type == "how_to":
                fact_sheet = format_fact_sheet(chunks, target_operation)
                is_valid = self._validate_response(formatted_answer, target_operation, fact_sheet)
                if not is_valid:
                    logger.warning(f"RAG Pipeline: LLM response failed validation in image query. Triggering safe fallback.")
                    formatted_answer = self._generate_fallback_response(target_operation, fact_sheet)

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
                "answer": formatted_answer,
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

    def _rerank_results(self, results: List[Any], user_query: str, top_k: int, target_operation: str = "None", category: str = "general") -> List[Any]:
        """
        Re-ranks retrieved vector chunks by boosting similarity scores for chunks
        matching the target operation and category.
        """
        if not results:
            return []
            
        query_lower = user_query.lower()
        
        for res in results:
            chunk = res.chunk
            meta = chunk.metadata or {}
            
            endpoint_name = (meta.get("endpoint_name") or meta.get("operation_id") or meta.get("request_name") or "").lower()
            chunk_category = (meta.get("category") or "").lower()
            http_method = (meta.get("http_method") or "").lower()
            url_path = (meta.get("url") or meta.get("api_path") or "").lower()
            
            normalized_endpoint = endpoint_name.replace(" ", "_").replace("-", "_")
            
            boost = 0.0
            
            # Category boost
            if chunk_category == category:
                boost += 0.20
                
            # Exact or partial operation name match
            if target_operation != "None":
                if normalized_endpoint == target_operation or target_operation in normalized_endpoint:
                    boost += 0.40
                elif is_operation_match(endpoint_name, target_operation):
                    boost += 0.40
                elif target_operation == "oauth_token" and any(k in normalized_endpoint for k in ["oauth", "ouath", "token"]):
                    boost += 0.40
                    
            # HTTP Method presence in query check
            if http_method and http_method in query_lower:
                boost += 0.10
                
            res.similarity_score += boost
            
        # Re-sort results descending by the boosted score
        reranked = sorted(results, key=lambda x: x.similarity_score, reverse=True)
        return reranked[:top_k]
