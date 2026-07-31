import json
import yaml
from typing import Tuple, List, Dict, Any
from core.models import Document, DocumentChunk
from services.parser.base import BaseParser
from core.exceptions import ParsingException


class OpenAPIParser(BaseParser):
    """
    Parser for OpenAPI/Swagger (JSON and YAML) documents.
    Generates structured, atomic Markdown chunks for each API endpoint.
    """

    def parse(self, file_content: bytes, filename: str) -> Tuple[Document, List[DocumentChunk]]:
        try:
            # 1. Deserialize the spec based on file content type (JSON vs YAML)
            content_str = file_content.decode("utf-8", errors="ignore")
            
            # Attempt to detect if it's JSON or YAML
            spec_data: Dict[str, Any] = {}
            is_yaml = False
            
            ext = self._get_extension(filename)
            if ext in (".yaml", ".yml"):
                is_yaml = True
            
            if is_yaml:
                try:
                    spec_data = yaml.safe_load(content_str)
                except yaml.YAMLError as ye:
                    raise ParsingException(filename, f"Invalid YAML formatting: {ye}")
            else:
                try:
                    spec_data = json.loads(content_str)
                except json.JSONDecodeError as je:
                    # Fallback check: try parsing as YAML in case of extension mismatch
                    try:
                        spec_data = yaml.safe_load(content_str)
                    except Exception:
                        raise ParsingException(filename, f"Invalid JSON formatting: {je}")

            if not isinstance(spec_data, dict):
                raise ParsingException(filename, "Specification root must be an object/dict")

            # Check if it actually looks like an OpenAPI spec
            is_openapi = "openapi" in spec_data or "swagger" in spec_data
            if not is_openapi:
                raise ParsingException(filename, "File is not a valid OpenAPI or Swagger specification")

            # 2. Extract global metadata
            info = spec_data.get("info", {})
            title = info.get("title", "Unknown API")
            version = info.get("version", "1.0.0")
            api_description = info.get("description", "")
            servers = [s.get("url") for s in spec_data.get("servers", []) if s.get("url")]
            base_url = servers[0] if servers else "N/A"

            doc = Document(filename=filename, file_type="openapi")
            doc.metadata.update({
                "title": title,
                "version": version,
                "base_url": base_url,
                "description": api_description[:200]
            })

            chunks: List[DocumentChunk] = []
            
            # 3. Create global info chunk
            global_info_content = (
                f"# API Specification: {title}\n"
                f"Version: {version}\n"
                f"Base URL: {base_url}\n"
                f"Description: {api_description}\n"
            )
            chunks.append(
                DocumentChunk(
                    document_id=doc.id,
                    content=self._normalize(global_info_content),
                    chunk_index=0,
                    metadata={"filename": filename, "type": "global_metadata"}
                )
            )

            # Extract component schemas to resolve inline definitions in endpoints (basic resolver helper)
            components = spec_data.get("components", {})
            schemas = components.get("schemas", {})

            # 4. Iterate paths and operations to make atomic chunks
            paths = spec_data.get("paths", {})
            for path, path_item in paths.items():
                if not isinstance(path_item, dict):
                    continue
                
                # Check methods (get, post, put, delete, patch, options, head)
                for method, operation in path_item.items():
                    if method.lower() not in ("get", "post", "put", "delete", "patch", "options", "head"):
                        continue
                    
                    if not isinstance(operation, dict):
                        continue

                    summary = operation.get("summary", "")
                    description = operation.get("description", "")
                    operation_id = operation.get("operationId", "N/A")
                    parameters = operation.get("parameters", [])
                    request_body = operation.get("requestBody", {})
                    responses = operation.get("responses", {})

                    # Synthesize markdown block representing this endpoint
                    endpoint_content = []
                    endpoint_content.append(f"## Endpoint: {method.upper()} {path}")
                    if summary:
                        endpoint_content.append(f"**Summary**: {summary}")
                    if description:
                        endpoint_content.append(f"**Description**: {description}")
                    endpoint_content.append(f"**Operation ID**: {operation_id}")

                    # Format parameters
                    if parameters:
                        endpoint_content.append("\n### Parameters:")
                        for param in parameters:
                            p_name = param.get("name", "")
                            p_in = param.get("in", "")
                            p_req = "Required" if param.get("required") else "Optional"
                            p_desc = param.get("description", "")
                            p_schema = param.get("schema", {})
                            p_type = p_schema.get("type", "string")
                            
                            endpoint_content.append(f"- `{p_name}` ({p_in}, {p_type}, {p_req}): {p_desc}")

                    # Format request body
                    if request_body and isinstance(request_body, dict):
                        endpoint_content.append("\n### Request Body:")
                        req_desc = request_body.get("description", "")
                        if req_desc:
                            endpoint_content.append(f"Description: {req_desc}")
                        
                        content = request_body.get("content", {})
                        for media_type, media_obj in content.items():
                            endpoint_content.append(f"- **Content-Type**: `{media_type}`")
                            schema_ref = media_obj.get("schema", {})
                            
                            # Basic resolution of schema reference
                            resolved_schema = self._resolve_schema(schema_ref, schemas)
                            schema_str = json.dumps(resolved_schema, indent=2)
                            endpoint_content.append(f"  **Schema**:\n```json\n{schema_str}\n```")

                    # Format responses
                    if responses:
                        endpoint_content.append("\n### Responses:")
                        for status, response in responses.items():
                            resp_desc = response.get("description", "")
                            endpoint_content.append(f"- **{status}**: {resp_desc}")
                            
                            resp_content = response.get("content", {}) if isinstance(response, dict) else {}
                            if resp_content:
                                for media_type, media_obj in resp_content.items():
                                    schema_ref = media_obj.get("schema", {})
                                    resolved_schema = self._resolve_schema(schema_ref, schemas)
                                    schema_str = json.dumps(resolved_schema, indent=2)
                                    endpoint_content.append(f"  - Content-Type: `{media_type}`\n    Schema:\n```json\n{schema_str}\n```")

                    # Compile and normalize
                    full_endpoint_text = "\n".join(endpoint_content)
                    
                    # Determine category from tags or path
                    tags = operation.get("tags", [])
                    category = tags[0] if tags else "General API Reference"
                    path_lower = path.lower()
                    if "auth" in path_lower or "token" in path_lower:
                        category = "Authentication"
                    elif "site" in path_lower:
                        category = "Sites"
                    elif "user" in path_lower or "role" in path_lower or "permission" in path_lower:
                        category = "User Management"
                    elif "access" in path_lower or "device" in path_lower or "reader" in path_lower:
                        category = "Access Control"
                    elif "card" in path_lower or "credential" in path_lower:
                        category = "Card Assignment"

                    # Determine authentication
                    security = operation.get("security", spec_data.get("security", []))
                    auth_type = "Bearer Token / API Key" if security else "None or Inherited"

                    chunk_metadata = {
                        "filename": filename,
                        "source_document": filename,
                        "type": "endpoint",
                        "chunk_type": "api_endpoint",
                        "document_type": "api_specification",
                        "endpoint_name": summary or operation_id or f"{method.upper()} {path}",
                        "http_method": method.upper(),
                        "url": path,
                        "api_path": path,
                        "authentication": auth_type,
                        "category": category,
                        "description": description or summary,
                        "notes": operation.get("description", "") if summary else "",
                        "headers": [p.get("name") for p in parameters if p.get("in") == "header"] if parameters else [],
                        "request_parameters": [
                            {"name": p.get("name"), "in": p.get("in"), "required": p.get("required")}
                            for p in parameters if p.get("in") in ("query", "path")
                        ] if parameters else [],
                        "operation_id": operation_id
                    }

                    chunks.append(
                        DocumentChunk(
                            document_id=doc.id,
                            content=self._normalize(full_endpoint_text),
                            chunk_index=len(chunks),
                            metadata=chunk_metadata
                        )
                    )

            # 5. Extract schemas as separate chunks (in case queries ask generally about schemas)
            if schemas:
                for schema_name, schema_val in schemas.items():
                    schema_text = (
                        f"## Component Schema: {schema_name}\n"
                        f"```json\n{json.dumps(schema_val, indent=2)}\n```"
                    )
                    chunks.append(
                        DocumentChunk(
                            document_id=doc.id,
                            content=self._normalize(schema_text),
                            chunk_index=len(chunks),
                            metadata={
                                "filename": filename,
                                "type": "schema",
                                "schema_name": schema_name
                            }
                        )
                    )

            return doc, chunks

        except ParsingException as pe:
            raise pe
        except Exception as e:
            self._handle_error(filename, e)

    def _resolve_schema(self, schema: Dict[str, Any], schemas: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """
        Recursively resolves references ($ref) in OpenAPI schemas to present inline schemas in chunks.
        """
        if depth > 5:  # Prevent infinite loops in circular references
            return {"type": "object", "description": "Circular reference detected"}

        if not isinstance(schema, dict):
            return schema

        if "$ref" in schema:
            ref_path = schema["$ref"]
            schema_name = ref_path.split("/")[-1]
            if schema_name in schemas:
                # Retrieve schema and recurse resolution
                resolved = schemas[schema_name]
                return self._resolve_schema(resolved, schemas, depth + 1)
            return {"type": "object", "description": f"Unresolved reference: {ref_path}"}

        # Resolve properties inside objects
        resolved_properties = {}
        if "properties" in schema and isinstance(schema["properties"], dict):
            for k, v in schema["properties"].items():
                resolved_properties[k] = self._resolve_schema(v, schemas, depth + 1)
            new_schema = schema.copy()
            new_schema["properties"] = resolved_properties
            return new_schema

        # Resolve array items
        if "items" in schema and isinstance(schema["items"], dict):
            new_schema = schema.copy()
            new_schema["items"] = self._resolve_schema(schema["items"], schemas, depth + 1)
            return new_schema

        return schema
