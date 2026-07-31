import json
from typing import Tuple, List, Dict, Any
from uuid import UUID
from core.models import Document, DocumentChunk
from services.parser.base import BaseParser
from core.exceptions import ParsingException


class PostmanCollectionParser(BaseParser):
    """
    Parser for Postman Collections (JSON).
    Recursively navigates collection folders and generates atomic Markdown chunks for each request.
    """

    def parse(self, file_content: bytes, filename: str) -> Tuple[Document, List[DocumentChunk]]:
        try:
            content_str = file_content.decode("utf-8", errors="ignore")
            
            try:
                collection_data = json.loads(content_str)
            except json.JSONDecodeError as je:
                raise ParsingException(filename, f"Invalid JSON formatting: {je}")

            if not isinstance(collection_data, dict):
                raise ParsingException(filename, "Collection root must be an object/dict")

            # Check for Postman info block
            info = collection_data.get("info", {})
            if not info or "schema" not in info:
                raise ParsingException(filename, "File is not a valid Postman Collection (missing 'info' or 'schema')")

            collection_name = info.get("name", "Unknown Postman Collection")
            collection_description = info.get("description", "")

            doc = Document(filename=filename, file_type="postman")
            doc.metadata.update({
                "collection_name": collection_name,
                "description": collection_description[:200]
            })

            chunks: List[DocumentChunk] = []

            # Create global collection info chunk
            global_content = (
                f"# Postman Collection: {collection_name}\n"
                f"Description: {collection_description}\n"
            )
            chunks.append(
                DocumentChunk(
                    document_id=doc.id,
                    content=self._normalize(global_content),
                    chunk_index=0,
                    metadata={"filename": filename, "type": "global_metadata"}
                )
            )

            # Traverse the nested items recursively
            items = collection_data.get("item", [])
            self._walk_items(items, [], chunks, doc.id, filename)

            # Re-index the chunks sequentially
            for index, chunk in enumerate(chunks):
                chunk.chunk_index = index

            return doc, chunks

        except ParsingException as pe:
            raise pe
        except Exception as e:
            self._handle_error(filename, e)

    def _walk_items(
        self, 
        items: List[Dict[str, Any]], 
        folder_path: List[str], 
        chunks: List[DocumentChunk], 
        doc_id: UUID, 
        filename: str
    ) -> None:
        """
        Recursively walk the items tree to extract folders and requests.
        """
        for item in items:
            if not isinstance(item, dict):
                continue

            name = item.get("name", "Unnamed Item")
            
            # If item contains a nested list of items, it's a folder/group
            if "item" in item and isinstance(item["item"], list):
                self._walk_items(item["item"], folder_path + [name], chunks, doc_id, filename)
            
            # If item contains a request object, it's an API request leaf
            elif "request" in item:
                request = item["request"]
                if isinstance(request, dict):
                    request_chunk = self._parse_request(item, request, folder_path, doc_id, filename)
                    if request_chunk:
                        chunks.append(request_chunk)

    def _parse_request(
        self, 
        item_obj: Dict[str, Any], 
        request: Dict[str, Any], 
        folder_path: List[str], 
        doc_id: UUID, 
        filename: str
    ) -> DocumentChunk:
        """
        Parse a single Postman Request and format it as a clean Markdown chunk.
        """
        name = item_obj.get("name", "Unnamed Request")
        method = request.get("method", "GET").upper()
        description = request.get("description", "")
        
        # Parse URL
        url_obj = request.get("url", "")
        raw_url = ""
        path_str = ""
        
        if isinstance(url_obj, dict):
            raw_url = url_obj.get("raw", "")
            path_segments = url_obj.get("path", [])
            # Reconstruct relative path
            if isinstance(path_segments, list):
                path_str = "/" + "/".join(str(s) for s in path_segments if s)
        elif isinstance(url_obj, str):
            raw_url = url_obj
            path_str = url_obj  # fallback

        # Format markdown content for this request
        req_content = []
        folder_context = " -> ".join(folder_path) if folder_path else "Root"
        
        req_content.append(f"## Request: {name}")
        req_content.append(f"**Folder Pathway**: `{folder_context}`")
        req_content.append(f"**HTTP Method**: `{method}`")
        req_content.append(f"**Target URL**: `{raw_url}`")
        
        if description:
            req_content.append(f"**Description**: {description}")

        # Extract headers
        headers = request.get("header", [])
        if headers:
            req_content.append("\n### Request Headers:")
            for h in headers:
                if isinstance(h, dict):
                    h_key = h.get("key", "")
                    h_val = h.get("value", "")
                    h_desc = h.get("description", "")
                    h_desc_str = f" - {h_desc}" if h_desc else ""
                    req_content.append(f"- `{h_key}`: `{h_val}`{h_desc_str}")

        # Extract URL variables & query parameters
        if isinstance(url_obj, dict):
            variables = url_obj.get("variable", [])
            if variables:
                req_content.append("\n### Path Variables:")
                for v in variables:
                    v_key = v.get("key", "")
                    v_val = v.get("value", "")
                    v_desc = v.get("description", "")
                    v_desc_str = f" - {v_desc}" if v_desc else ""
                    req_content.append(f"- `{v_key}`: `{v_val}`{v_desc_str}")

            queries = url_obj.get("query", [])
            if queries:
                req_content.append("\n### Query Parameters:")
                for q in queries:
                    q_key = q.get("key", "")
                    q_val = q.get("value", "")
                    q_desc = q.get("description", "")
                    q_desc_str = f" - {q_desc}" if q_desc else ""
                    req_content.append(f"- `{q_key}`: `{q_val}`{q_desc_str}")

        # Extract body payloads
        body_obj = request.get("body", {})
        if body_obj and isinstance(body_obj, dict):
            body_mode = body_obj.get("mode", "")
            if body_mode:
                req_content.append(f"\n### Request Body ({body_mode}):")
                if body_mode == "raw":
                    raw_body = body_obj.get("raw", "")
                    # Indent raw body code blocks
                    req_content.append(f"```json\n{raw_body}\n```")
                elif body_mode == "urlencoded":
                    params = body_obj.get("urlencoded", [])
                    for p in params:
                        p_key = p.get("key", "")
                        p_val = p.get("value", "")
                        req_content.append(f"- `{p_key}`: `{p_val}`")
                elif body_mode == "formdata":
                    params = body_obj.get("formdata", [])
                    for p in params:
                        p_key = p.get("key", "")
                        p_val = p.get("value", "")
                        p_type = p.get("type", "text")
                        req_content.append(f"- `{p_key}` ({p_type}): `{p_val}`")

        # Extract responses if present
        responses = item_obj.get("response", [])
        if responses and isinstance(responses, list):
            req_content.append("\n### Mock Responses / Saved Examples:")
            for r in responses:
                if not isinstance(r, dict):
                    continue
                r_name = r.get("name", "Example Response")
                r_code = r.get("code", 200)
                r_status = r.get("status", "OK")
                r_body = r.get("body", "")
                
                req_content.append(f"\n#### Example: {r_name} (Status: {r_code} {r_status})")
                if r_body:
                    req_content.append(f"```json\n{r_body}\n```")

        # Compile text chunk
        full_text = "\n".join(req_content)

        # Determine category based on folders, request name, and path keywords
        category = "General API Reference"
        folder_str = " -> ".join(folder_path).lower() if folder_path else ""
        request_identifier = f"{name} {raw_url} {path_str} {folder_str}".lower()
        
        if any(k in request_identifier for k in ["auth", "login", "oauth", "ouath", "token"]):
            category = "Authentication"
        elif "site" in request_identifier:
            category = "Sites"
        elif any(k in request_identifier for k in ["user", "role", "permission"]):
            category = "User Management"
        elif any(k in request_identifier for k in ["access", "device", "reader"]):
            category = "Access Control"
        elif any(k in request_identifier for k in ["card", "credential"]):
            category = "Card Assignment"
        elif folder_path:
            category = folder_path[0]

        # Determine authentication
        auth_obj = request.get("auth", {})
        auth_type = "Bearer Token / API Key" if auth_obj else "None or Inherited"

        chunk_metadata = {
            "filename": filename,
            "source_document": filename,
            "type": "postman_request",
            "chunk_type": "api_endpoint",
            "document_type": "api_specification",
            "endpoint_name": name,
            "request_name": name,
            "http_method": method,
            "url": raw_url or path_str,
            "api_path": path_str,
            "authentication": auth_type,
            "category": category,
            "description": description,
            "notes": f"Folder pathway: {' -> '.join(folder_path)}" if folder_path else "",
            "headers": [h.get("key") for h in headers if isinstance(h, dict) and h.get("key")] if headers else [],
            "request_parameters": [
                {"name": q.get("key"), "in": "query"}
                for q in (url_obj.get("query", []) if isinstance(url_obj, dict) else [])
                if isinstance(q, dict) and q.get("key")
            ],
            "folder_pathway": folder_path
        }

        return DocumentChunk(
            document_id=doc_id,
            content=self._normalize(full_text),
            chunk_index=0,  # will be re-indexed in walk
            metadata=chunk_metadata
        )
