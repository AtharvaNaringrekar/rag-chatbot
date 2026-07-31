import re
from typing import Dict, Any, List


def extract_api_metadata_from_text(text: str) -> Dict[str, Any]:
    """
    Analyzes API documentation text and extracts structured metadata fields
    to assist with semantic re-ranking and retrieval search.
    """
    metadata = {}
    text_lower = text.lower()
    
    # Default tags
    metadata["document_type"] = "api_documentation"
    metadata["chunk_type"] = "text_guide"
    metadata["authentication"] = "None or Inherited"
    metadata["category"] = "General API Reference"

    # 1. Determine HTTP Method & URL path
    methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
    found_method = None
    
    # Look for "POST /api/v1/users" or similar combinations
    method_url_match = re.search(
        r'\b(GET|POST|PUT|DELETE|PATCH)\b\s+([\w\-\./\{\}\?&=:%]+)', 
        text, 
        re.IGNORECASE
    )
    if method_url_match:
        found_method = method_url_match.group(1).upper()
        metadata["http_method"] = found_method
        metadata["url"] = method_url_match.group(2)
        metadata["chunk_type"] = "api_endpoint"
    else:
        # Fallback search for method keyword at the start of lines
        for m in methods:
            if re.search(r'\b' + m + r'\b', text[:300], re.IGNORECASE):
                found_method = m
                metadata["http_method"] = m
                metadata["chunk_type"] = "api_endpoint"
                break
                
    # 2. Extract URL path if not matched already
    if "url" not in metadata:
        url_match = re.search(r'(/(?:api|oauth|v\d+)/[\w\-\./\{\}]+)', text)
        if url_match:
            metadata["url"] = url_match.group(1)
            metadata["chunk_type"] = "api_endpoint"

    # 3. Determine Endpoint Name / Title
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        first_line = lines[0]
        # Clean markdown characters like ## or **
        endpoint_name = re.sub(r'^[#*\-\s]+', '', first_line).strip()
        endpoint_name = re.sub(r'[#*\-:]+$', '', endpoint_name).strip()
        
        # Clean HTTP Method and URL patterns out of the title to keep it readable
        for m in methods:
            endpoint_name = re.sub(r'\b' + m + r'\b', '', endpoint_name, flags=re.IGNORECASE).strip()
        endpoint_name = re.sub(r'/(?:api|v\d+|oauth)/[\w\-\./\{\}]+', '', endpoint_name, flags=re.IGNORECASE).strip()
        endpoint_name = endpoint_name.strip("*-: ").strip()
        
        if endpoint_name and len(endpoint_name) < 80:
            metadata["endpoint_name"] = endpoint_name

    # 4. Extract Category based on keyword context matching
    if any(k in text_lower for k in ["oauth", "login", "auth", "token", "sign-in", "session", "credentials"]):
        metadata["category"] = "Authentication"
    elif any(k in text_lower for k in ["site", "organization", "org"]):
        metadata["category"] = "Sites"
    elif any(k in text_lower for k in ["user", "role", "permission", "member"]):
        metadata["category"] = "User Management"
    elif any(k in text_lower for k in ["access point", "reader", "device", "door", "lock", "gateway", "access-point"]):
        metadata["category"] = "Access Control"
    elif any(k in text_lower for k in ["card", "credential", "badge", "tag", "assign", "unassign"]):
        metadata["category"] = "Card Assignment"
    elif any(k in text_lower for k in ["workflow", "example", "scenario", "flow"]):
        metadata["category"] = "Workflow Examples"

    # 5. Extract Authentication requirements
    auth_patterns = [
        r'(?:bearer|jwt|api[-_]key|auth[-_]token|token)\b',
        r'authorization\s*:\s*\w+'
    ]
    has_auth = False
    for pat in auth_patterns:
        if re.search(pat, text_lower):
            has_auth = True
            break
    if has_auth:
        metadata["authentication"] = "Bearer Token / API Key"

    # 6. Extract Headers
    headers = []
    header_matches = re.findall(
        r'\b([\w-]+)\s*:\s*(?:application/json|Bearer|X-[\w-]+|[\w/]+)', 
        text
    )
    if header_matches:
        for h in header_matches:
            if h.lower() not in ("http", "https", "localhost"):
                headers.append(h)
    if headers:
        metadata["headers"] = list(set(headers))

    # 7. Extract JSON Request Body and Response Examples
    json_blocks = re.findall(r'```json\n(.*?)\n```', text, re.DOTALL)
    if json_blocks:
        metadata["request_body"] = json_blocks[0].strip()
        if len(json_blocks) > 1:
            metadata["response"] = json_blocks[1].strip()
    else:
        # Fallback to general curly braces matching
        json_obj_match = re.search(r'(\{\s*"[\s\S]*?"\s*\})', text)
        if json_obj_match:
            metadata["request_body"] = json_obj_match.group(1).strip()

    return metadata
