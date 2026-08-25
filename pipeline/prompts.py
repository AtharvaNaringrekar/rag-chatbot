import re
from typing import List
from core.models import DocumentChunk

# Strict System Prompt prefix (retained for test compatibility)
SYSTEM_PROMPT_PREFIX = (
    "You are a Spintly API Support Engineer. Answer the user query using ONLY the provided documentation context. "
    "Be direct, concise, and professional. Do NOT speak like a search engine or mention documents.\n\n"
    "CRITICAL RULES & CPU LATENCY GUARDRAILS:\n"
    "1. General Fallback: If information is missing, respond EXACTLY: \"I could not find this information in the available Spintly documentation.\" "
    "(Ensure the exact fallback phrase \"I couldn't find that information in the uploaded documentation.\" is also matched.)\n"
    "2. Troubleshooting Fallback: If troubleshooting steps are missing, respond EXACTLY: \"The retrieved Spintly documentation does not provide specific troubleshooting steps for this issue.\"\n"
    "3. Zero Hallucinations / No Inference: Do not invent endpoints, HTTP methods, parameters, request bodies, response fields, status codes, headers, or authentication schemes. Base answers strictly on the retrieved context. Never mix context with pre-trained LLM knowledge. If required information is not present, use the configured fallback.\n"
    "4. Hide Credentials: Replace API tokens/secrets with placeholders like <ACCESS_TOKEN>.\n"
    "5. Hide Internal Terminology: Do NOT display document names, page numbers, sections, context block IDs, or metadata headers. Never mention the terms 'Document Segment', 'Context Block', 'segment', 'context', 'metadata', or 'database'. Speak natively.\n"
    "6. Format: Use structured markdown (code blocks, headers, lists). Omit empty sections or placeholders entirely.\n"
    "7. Question-Specific: Answer the user's exact question directly and ignore unrelated APIs or generic filler. Use simple, beginner-friendly language.\n"
    "8. Strict Operation Isolation: Identify the target operation (e.g. create_user, update_user, deactivate_user, activate_user, delete_user, create_meeting, update_meeting, delete_meeting) and prioritize document segments matching that operation. TARGET OPERATION is authoritative for procedural API questions. Use only information belonging to the TARGET OPERATION. Ignore unrelated API operations even if they contain words similar to the user's question. Never combine request bodies, endpoints, HTTP methods, or instructions from different operations.\n"
    "9. Exact API Fact Preservation: Every HTTP method, endpoint URL, JSON key, parameter name, camelCase field name, header, and exact placeholder (like <CLIENT_ID>, <ACCESS_TOKEN>, <USER_ID>) must be preserved exactly as documented. Do NOT convert placeholders to lowercase/descriptive words (do not change <CLIENT_ID> to your-client-id).\n"
    "10. Postman Fidelity & Request Body Priority: If the query seeks steps, format using a numbered '## Steps' section. Use ONLY the documented Postman steps from the target segment. Do NOT invent web browser instructions. If a request body is documented, render the complete and exact JSON code block inline in the steps (never truncate, abbreviate, or omit fields; preserve all keys, camelCase, types, and documented placeholder/example values). End the steps with a 'Click Send' action. API HOW-TO questions are exempt from normal conciseness/short-answer/brevity constraints; completeness has priority over brevity, and the response must never be truncated, cut off, or summarized.\n\n"
    "RESPONSE TEMPLATE & ADAPTIVE STRUCTURES:\n"
    "Before generating, classify the user query into the target category and format your response using ONLY the sections specified in that category template:\n\n"
)

TEMPLATE_DEFINITION = (
    "DEFINITION / CONCEPT QUESTIONS (e.g., 'What is homeSiteId?', 'Explain OAuth'):\n"
    "Format your response as a single, direct, concise paragraph (max 60 words) explaining the concept. "
    "Do NOT include headings or unrelated API procedures."
)

TEMPLATE_SHORT_FACT = (
    "SHORT FACT QUESTIONS (e.g., 'What is the endpoint for Get Access Points?'):\n"
    "Format your response as a direct, concise answer showing the HTTP Method, URL, and Authentication if relevant. "
    "Do NOT include headings, bullet points, or step-by-step instructions unless explicitly asked."
)

TEMPLATE_HOW_TO = (
    "## Step-by-Step Instructions\n"
    "[Numbered list of instructions to complete the task]\n\n"
    "## Example Request\n"
    "[Markdown code block of the request, MUST include if available in documentation]\n\n"
    "## Example Response\n"
    "[Markdown code block of the response, MUST include if available in documentation]"
)

TEMPLATE_API_ENDPOINT = (
    "API ENDPOINT QUESTIONS (e.g., 'Explain Create User API'):\n"
    "## API Purpose\n"
    "[Brief purpose]\n\n"
    "## Endpoint Details\n"
    "- **HTTP Method**: [Method in uppercase]\n"
    "- **URL**: [URL path]\n"
    "- **Authentication**: [Bearer Token or None]\n\n"
    "## Request Body\n"
    "[JSON request body block, only if documented]\n\n"
    "## Success Response\n"
    "[Success status and body details, only if documented]"
)

TEMPLATE_TROUBLESHOOTING = (
    "TROUBLESHOOTING QUESTIONS (e.g., 'Why am I getting 401?'):\n"
    "1. Documented Cause: [Likely cause from context]\n"
    "2. Documented Check: [How to verify from context]\n"
    "3. Documented Solution: [Resolution steps from context]"
)

TEMPLATE_COMPARISON = (
    "COMPARISON QUESTIONS:\n"
    "## Overview\n"
    "[Short introduction paragraph]\n\n"
    "## Comparison Details\n"
    "| Feature/Aspect | [Concept 1] | [Concept 2] |\n"
    "| :--- | :--- | :--- |\n"
    "| [Comparison Row 1] | [Value 1] | [Value 2] |\n\n"
    "## Key Differences\n"
    "[Bulleted list of key comparison differences]\n"
)

# Unified Grounded System Prompt
SYSTEM_PROMPT = (
    "You are a Spintly API Technical Support Engineer.\n\n"
    "Answer the user's question using ONLY the provided verified API information.\n\n"
    "Preserve API facts exactly.\n\n"
    "Do not invent or modify endpoints, HTTP methods, headers, JSON keys, request bodies, parameters, or documented example values.\n\n"
    "Do not mix operations.\n\n"
    "For procedural/API answers, you MUST structure your response with the following headings if the information is available:\n"
    "## Step-by-Step Instructions\n"
    "[Numbered list of instructions to complete the task]\n\n"
    "## Example Request\n"
    "[Markdown code block of the JSON request body payload, MUST include if available in documentation]\n\n"
    "## Example Response\n"
    "[Markdown code block of the response JSON payload, MUST include if available in documentation]\n\n"
    "If a complete request body is provided, include the complete request body.\n\n"
    "Do not mention internal RAG terminology.\n\n"
    "If the required information is unavailable, respond:\n"
    "'I could not find this information in the available Spintly documentation.'"
)


# User Prompt Template
USER_PROMPT_TEMPLATE = (
    "### TECHNICAL DOCUMENTATION CONTEXT:\n"
    "{context_text}\n\n"
    "TARGET OPERATION: {target_operation}\n\n"
    "TARGET QUESTION: {user_question}\n\n"
    "### SUPPORT ENGINEER RESPONSE:"
)


# Image Diagnostic User Prompt Template
IMAGE_USER_PROMPT_TEMPLATE = (
    "### TECHNICAL DOCUMENTATION CONTEXT:\n"
    "{context_text}\n\n"
    "### SCREENSHOT DIAGNOSTIC DETAILS:\n"
    "- Visual Description: {vision_description}\n"
    "- Extracted Text/Logs:\n"
    "```\n"
    "{ocr_text}\n"
    "```\n\n"
    "TARGET OPERATION: {target_operation}\n\n"
    "TARGET QUESTION: {user_question}\n\n"
    "### SUPPORT ENGINEER RESPONSE:"
)


# Authoritative, 100% verified source facts dictionary
VERIFIED_API_SPECS = {
    "oauth_token": {
        "method": "POST",
        "url": "https://qlwtofmb58.execute-api.ap-south-1.amazonaws.com/prod/saamsIdm/oauth/token",
        "auth": "No Auth",
        "headers": ["Content-Type: application/json"],
        "body": {
            "clientId": "<CLIENT_ID>",
            "clientSecret": "<CLIENT_SECRET>",
            "grantType": "urn:ietf:params:oauth:grant-type:client-credentials"
        },
        "purpose": "Generate OAuth access token using client credentials.",
        "definition_overview": "An OAuth access token in Spintly is a temporary Bearer token generated by the Spintly OAuth token endpoint. It is obtained using the documented client credentials flow (No Auth, passing client credentials in request body) and is used for authorizing subsequent Spintly API calls.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to POST.",
            "Enter the endpoint: https://qlwtofmb58.execute-api.ap-south-1.amazonaws.com/prod/saamsIdm/oauth/token",
            "Under the Authorization tab, set the type to No Auth.",
            "Under Headers, set Content-Type to application/json.",
            "Under Body, select raw and JSON, and enter the client credentials payload.",
            "Click Send."
        ]
    },
    "create_user": {
        "method": "POST",
        "url": "https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/create",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "name": "j1ane doe",
            "phone": "",
            "roles": [24306],
            "adminOfSites": [None],
            "devicelock": True,
            "accessData": {
                "mobile": True,
                "card": True,
                "fingerprint": True,
                "remoteAccess": False,
                "clickToAccessRange": 1,
                "accessExpiresAt": None,
                "tapToAccess": True,
                "accessPoints": [],
                "gps": True,
                "deviceLock": True,
                "faceAccess": True
            },
            "homeSiteId": 13480,
            "terms": [],
            "employeeCode": "10111",
            "reportingTo": None,
            "probationPeriod": None,
            "joiningDate": None,
            "credentialId": None
        },
        "purpose": "Create a new user with specific role and access data inside Spintly.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to POST.",
            "Enter the endpoint: https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/create",
            "Under Authorization, set the type to Bearer Token and insert the JWT access token.",
            "Under Headers, ensure Content-Type is set to application/json.",
            "Under Body, select raw and JSON, and enter the request body payload.",
            "Click Send."
        ]
    },
    "update_user": {
        "method": "PATCH",
        "url": "https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/update",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "users": [
                {
                    "id": 1197835,
                    "accessExpiresAt": None,
                    "employeeCode": "101",
                    "name": "jane doe11",
                    "reportingTo": None,
                    "homeSiteId": 13480,
                    "roles": [24306],
                    "terms": [],
                    "joiningDate": None,
                    "attributes": []
                }
            ]
        },
        "purpose": "Update user profiles (e.g. name, role, employee code) in bulk.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to PATCH.",
            "Enter the endpoint: https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/update",
            "Under Authorization, select Bearer Token and paste the JWT token.",
            "Under Headers, set Content-Type to application/json.",
            "Under Body, select raw and JSON, and input the users bulk update payload.",
            "Click Send."
        ]
    },
    "deactivate_user": {
        "method": "PATCH",
        "url": "https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/update",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "users": [
                {
                    "id": 1197835,
                    "deactivateUser": True
                }
            ]
        },
        "purpose": "Deactivate an active user by setting deactivateUser to true in bulk.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to PATCH.",
            "Enter the endpoint: https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/update",
            "Configure Authorization as Bearer Token.",
            "Under Headers, add Content-Type: application/json.",
            "Under Body, select raw and JSON, and enter the deactivation payload.",
            "Click Send."
        ]
    },
    "activate_user": {
        "method": "PATCH",
        "url": "https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/update",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "users": [
                {
                    "id": 1197835,
                    "deactivateUser": False,
                    "accessExpiresAt": None,
                    "employeeCode": "",
                    "name": "john",
                    "reportingTo": "",
                    "homeSiteId": 13480,
                    "adminOfSites": [],
                    "roles": [24306],
                    "terms": [],
                    "accessPoints": [],
                    "joiningDate": "2024-04-05",
                    "attributes": [],
                    "accessData": {
                        "gps": False,
                        "accessExpiresAt": None,
                        "accessPoints": [],
                        "mobile": True
                    }
                }
            ]
        },
        "purpose": "Re-activate a deactivated user by setting deactivateUser to false and specifying their access profile.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to PATCH.",
            "Enter the endpoint: https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/update",
            "Configure Authorization as Bearer Token.",
            "Under Headers, add Content-Type: application/json.",
            "Under Body, select raw and JSON, and enter the activation payload.",
            "Click Send."
        ]
    },
    "delete_user": {
        "method": "POST",
        "url": "https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/delete",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "userIds": [1197835]
        },
        "purpose": "Permanently delete specified users from the organisation.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to POST.",
            "Enter the endpoint: https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users/delete",
            "Configure Authorization as Bearer Token.",
            "Under Headers, add Content-Type: application/json.",
            "Under Body, select raw and JSON, and enter the user IDs payload.",
            "Click Send."
        ]
    },
    "grant_access_point_permissions": {
        "method": "PATCH",
        "url": "https://saams.api.spintly.com/accessManagementV3/integrator/v1/organisations/{organisationId}/accessPoint/{accessPointId}/users/permissions",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "permissionsToAdd": [1197306],
            "permissionsToRemove": [],
            "pendingPermissionsToRemove": []
        },
        "purpose": "Grant access point permissions to a user by adding their ID to permissionsToAdd.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to PATCH.",
            "Enter the endpoint: https://saams.api.spintly.com/accessManagementV3/integrator/v1/organisations/{organisationId}/accessPoint/{accessPointId}/users/permissions",
            "Configure Authorization as Bearer Token.",
            "Under Headers, set Content-Type to application/json.",
            "Under Body, select raw and JSON, and enter the permissions payload.",
            "Click Send."
        ]
    },
    "revoke_access_point_permissions": {
        "method": "PATCH",
        "url": "https://saams.api.spintly.com/accessManagementV3/integrator/v1/organisations/{organisationId}/accessPoint/{accessPointId}/users/permissions",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "permissionsToAdd": [],
            "permissionsToRemove": [1197306],
            "pendingPermissionsToRemove": []
        },
        "purpose": "Revoke access point permissions from a user by adding their ID to permissionsToRemove.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to PATCH.",
            "Enter the endpoint: https://saams.api.spintly.com/accessManagementV3/integrator/v1/organisations/{organisationId}/accessPoint/{accessPointId}/users/permissions",
            "Configure Authorization as Bearer Token.",
            "Under Headers, set Content-Type to application/json.",
            "Under Body, select raw and JSON, and enter the permissions payload.",
            "Click Send."
        ]
    },
    "get_user_permissions": {
        "method": "POST",
        "url": "https://saams.api.spintly.com/accessManagementV3/integrator/v1/organisations/{organisationId}/users/{userId}/permissions",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "sites": []
        },
        "purpose": "Retrieve permissions assigned to a user.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to POST.",
            "Enter the endpoint: https://saams.api.spintly.com/accessManagementV3/integrator/v1/organisations/{organisationId}/users/{userId}/permissions",
            "Configure Authorization as Bearer Token.",
            "Under Headers, set Content-Type to application/json.",
            "Under Body, select raw and JSON, and enter the request body sites array.",
            "Click Send."
        ]
    },
    "get_roles": {
        "method": "GET",
        "url": "https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/formData?roles=roles",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": None,
        "purpose": "Fetch all roles available within the organisation.",
        "definition_overview": "Roles define access categories (such as Employee, super_admin, site_admin, end_user) configured within the Spintly organisation that must be assigned to users.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to GET.",
            "Enter the endpoint: https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/formData?roles=roles",
            "Configure Authorization as Bearer Token.",
            "Click Send."
        ]
    },
    "get_sites": {
        "method": "POST",
        "url": "https://saams.api.spintly.com/organisationManagement/v2/integrator/organisations/{organisationId}/sites",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "pagination": {
                "page": 1,
                "perPage": 40
            }
        },
        "purpose": "Retrieve paginated sites configured for the organisation.",
        "definition_overview": "A homeSiteId represents the unique identifier of a physical site (facility, location, or office building, e.g., Site A Goa) configured within the Spintly organization. It is retrieved using the Get Sites API and must be specified when creating or updating users.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to POST.",
            "Enter the endpoint: https://saams.api.spintly.com/organisationManagement/v2/integrator/organisations/{organisationId}/sites",
            "Configure Authorization as Bearer Token.",
            "Under Headers, set Content-Type to application/json.",
            "Under Body, select raw and JSON, and enter the pagination parameters.",
            "Click Send."
        ]
    },
    "get_access_points": {
        "method": "GET",
        "url": "https://saams.api.spintly.com/accessManagementV3/integrator/v1/organisations/{organisationId}/getAccessPointList",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": None,
        "purpose": "Retrieve list of all access points / doors for the organisation.",
        "definition_overview": "Access points represent the physical doors, barriers, readers, or access controllers configured for an organisation within Spintly. Users are granted or revoked permissions to access these points.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to GET.",
            "Enter the endpoint: https://saams.api.spintly.com/accessManagementV3/integrator/v1/organisations/{organisationId}/getAccessPointList",
            "Configure Authorization as Bearer Token.",
            "Click Send."
        ]
    },
    "fetch_all_organisation_users": {
        "method": "POST",
        "url": "https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "pagination": {
                "page": 1,
                "perPage": 25,
                "currentPage": 1
            },
            "filters": {
                "createdOn": None,
                "userType": ["active"],
                "accessExpiresAt": None,
                "s": {
                    "email": ""
                },
                "terms": [],
                "sites": []
            }
        },
        "purpose": "Retrieve a paginated and filtered list of all organisation users.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to POST.",
            "Enter the endpoint: https://saams.api.spintly.com/userManagement/integrator/v1/organisations/{organisationId}/users",
            "Configure Authorization as Bearer Token.",
            "Under Headers, set Content-Type to application/json.",
            "Under Body, select raw and JSON, and enter the pagination and filtering parameters.",
            "Click Send."
        ]
    },
    "create_meeting": {
        "method": "POST",
        "url": "https://saams.api.spintly.com/visitorManagementV3/integrator/v1/organisations/{organisationId}/meeting",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "userId": "1190834",
            "startTime": 1781883600,
            "endTime": 1781942400,
            "permissionsToAdd": [23242]
        },
        "purpose": "Schedule/create a visitor management QR meeting.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to POST.",
            "Enter the endpoint: https://saams.api.spintly.com/visitorManagementV3/integrator/v1/organisations/{organisationId}/meeting",
            "Configure Authorization as Bearer Token.",
            "Under Headers, set Content-Type to application/json.",
            "Under Body, select raw and JSON, and enter the meeting request payload.",
            "Click Send."
        ]
    },
    "update_meeting": {
        "method": "PATCH",
        "url": "https://saams.api.spintly.com/visitorManagementV3/integrator/v1/organisations/{organisationId}/meeting/{meetingId}/update",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": {
            "startTime": 1774009800,
            "endTime": 1774012500,
            "permissionsToAdd": [23242],
            "permissionsToRemove": []
        },
        "purpose": "Update details of an existing VMS meeting.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to PATCH.",
            "Enter the endpoint: https://saams.api.spintly.com/visitorManagementV3/integrator/v1/organisations/{organisationId}/meeting/{meetingId}/update",
            "Configure Authorization as Bearer Token.",
            "Under Headers, set Content-Type to application/json.",
            "Under Body, select raw and JSON, and enter the update parameters payload.",
            "Click Send."
        ]
    },
    "delete_meeting": {
        "method": "DELETE",
        "url": "https://saams.api.spintly.com/visitorManagementV3/integrator/v1/organisations/{organisationId}/meeting/{meetingId}",
        "auth": "Bearer Token",
        "headers": ["Content-Type: application/json"],
        "body": None,
        "purpose": "Delete/cancel an existing VMS meeting.",
        "postman_steps": [
            "Open Postman and create a new request.",
            "Set the HTTP method to DELETE.",
            "Enter the endpoint: https://saams.api.spintly.com/visitorManagementV3/integrator/v1/organisations/{organisationId}/meeting/{meetingId}",
            "Configure Authorization as Bearer Token.",
            "Click Send."
        ]
    }
}


def format_context(chunks: List[DocumentChunk], include_metadata: bool = True) -> str:
    """
    Format a list of retrieved DocumentChunks into a clean, structured context block
    that the LLM can easily parse.
    """
    if not chunks:
        return "No documentation context available."
        
    context_blocks = []
    for idx, chunk in enumerate(chunks):
        header = f"--- Document Segment {idx + 1} ---"
        meta = chunk.metadata or {}
        
        if include_metadata:
            desc_parts = []
            
            filename = meta.get("filename", "unknown_file")
            desc_parts.append(f"Source: {filename}")
            
            endpoint_name = meta.get("endpoint_name") or meta.get("operation_id")
            if endpoint_name:
                desc_parts.append(f"Operation: {endpoint_name}")
                
            http_method = meta.get("http_method")
            if http_method:
                desc_parts.append(f"HTTP Method: {http_method}")
                
            url_path = meta.get("url") or meta.get("api_path")
            if url_path:
                desc_parts.append(f"Path: {url_path}")
                
            category = meta.get("category")
            if category:
                desc_parts.append(f"Category: {category}")
                
            auth_req = meta.get("authentication")
            if auth_req:
                desc_parts.append(f"Auth: {auth_req}")
                
            page = meta.get("page_number")
            if page:
                desc_parts.append(f"Page: {page}")
                
            section = meta.get("section")
            if section:
                desc_parts.append(f"Section: {section}")
                
            meta_text = "\n".join(desc_parts)
            block = f"{header}\n{meta_text}\n\n{chunk.content}"
        else:
            block = f"{header}\n{chunk.content}"
            
        context_blocks.append(block)
        
    return "\n\n".join(context_blocks)


def is_operation_match(endpoint_name: str, target_operation: str) -> bool:
    """
    Checks if a chunk's operation/endpoint name matches the target_operation.
    """
    endpoint_lower = endpoint_name.lower().replace("_", " ").replace("-", " ")
    op_lower = target_operation.lower().replace("_", " ").replace("-", " ")
    
    op_words = op_lower.split()
    if all(w in endpoint_lower for w in op_words):
        return True
        
    if target_operation == "oauth_token":
        return any(k in endpoint_lower for k in ["oauth", "access token", "token"])
    if target_operation == "deactivate_user":
        return "deactivate" in endpoint_lower or "suspend" in endpoint_lower or "disable" in endpoint_lower
    if target_operation == "activate_user":
        return "activate" in endpoint_lower or "enable" in endpoint_lower
    if target_operation == "delete_user":
        return "delete" in endpoint_lower or "remove" in endpoint_lower
    if target_operation == "create_user":
        return "create" in endpoint_lower or "add" in endpoint_lower or "register" in endpoint_lower or "onboard" in endpoint_lower
    if target_operation == "update_user":
        return "update" in endpoint_lower or "modify" in endpoint_lower or "edit" in endpoint_lower or "patch" in endpoint_lower
        
    return False


def format_fact_sheet(chunks: List[DocumentChunk], target_operation: str, query_type: str = "None") -> str:
    """
    Extracts and builds a clean, structured fact sheet for the target operation
    using the authoritative VERIFIED_API_SPECS database.
    """
    if target_operation in VERIFIED_API_SPECS:
        spec = VERIFIED_API_SPECS[target_operation]
        
        if query_type == "definition":
            fact_parts = [f"### SYSTEM FACT SHEET FOR OPERATION: {target_operation}"]
            fact_parts.append(f"- **Definition / Overview**: {spec.get('definition_overview', spec['purpose'])}")
            fact_parts.append(f"- **Purpose**: {spec['purpose']}")
            return "\n".join(fact_parts)
            
        import json
        body_str = ""
        if spec["body"] is not None:
            body_str = json.dumps(spec["body"], indent=2)
            
        fact_parts = [f"### SYSTEM FACT SHEET FOR OPERATION: {target_operation}"]
        fact_parts.append(f"- **Purpose**: {spec['purpose']}")
        fact_parts.append(f"- **HTTP Method**: {spec['method']}")
        fact_parts.append(f"- **Endpoint**: {spec['url']}")
        fact_parts.append(f"- **Authorization**: {spec['auth']}")
        
        if spec["headers"]:
            fact_parts.append("- **Required Headers**:")
            for h in spec["headers"]:
                fact_parts.append(f"  - {h}")
                
        if body_str:
            fact_parts.append(f"- **JSON Request Body**:\n```json\n{body_str}\n```")
            
        fact_parts.append("- **Postman Steps**:")
        for idx, step in enumerate(spec["postman_steps"]):
            fact_parts.append(f"  {idx+1}. {step}")
            
        return "\n".join(fact_parts)

    # Dynamic extraction fallback (for safety or non-core APIs)
    method = None
    url = None
    auth = None
    headers = []
    body = None
    postman_steps = []
    purpose = None
    
    for chunk in chunks:
        meta = chunk.metadata or {}
        content = chunk.content
        
        if not method:
            method = meta.get("http_method")
        if not url:
            url = meta.get("url") or meta.get("api_path")
        if not auth:
            auth = meta.get("authentication")
            
        chunk_headers = meta.get("headers")
        if chunk_headers and isinstance(chunk_headers, list):
            for h in chunk_headers:
                if h not in headers:
                    headers.append(h)
                    
        header_matches = re.findall(r"- `(.*?)`: `(.*?)`", content)
        for h_key, h_val in header_matches:
            h_str = f"{h_key}: {h_val}"
            if h_str not in headers:
                headers.append(h_str)
                
        if not purpose:
            purpose_match = re.search(r"Purpose\s*\n(.*?)(?=\n\n|\n[A-Z]|$)", content, re.DOTALL)
            if purpose_match:
                purpose = purpose_match.group(1).strip()
                
        json_blocks = re.findall(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
        if json_blocks:
            for j_block in json_blocks:
                j_block = j_block.strip()
                if any(k in j_block for k in ["userId", "clientId", "permissions", "deactivateUser", "meeting", "title"]):
                    j_block = re.sub(r'"clientId"\s*:\s*".*?"', '"clientId": "<CLIENT_ID>"', j_block)
                    j_block = re.sub(r'"clientSecret"\s*:\s*".*?"', '"clientSecret": "<CLIENT_SECRET>"', j_block)
                    body = j_block
                    break
            if not body:
                body = json_blocks[0].strip()
                
        if "Postman Verification Steps" in content or "postman steps" in content.lower():
            steps_match = re.search(r"(?:Postman Verification Steps|Postman Steps)\s*\n(.*?)($|\n\n|\n---)", content, re.DOTALL | re.IGNORECASE)
            if steps_match:
                steps_content = steps_match.group(1).strip()
                steps = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|$)", steps_content, re.DOTALL)
                if steps:
                    postman_steps = [s.strip() for s in steps]
                    
    headers_list_str = []
    for h in headers:
        if h not in headers_list_str:
            headers_list_str.append(h)
            
    fact_parts = [f"### SYSTEM FACT SHEET FOR OPERATION: {target_operation}"]
    if purpose:
        fact_parts.append(f"- **Purpose**: {purpose}")
    if method:
        fact_parts.append(f"- **HTTP Method**: {method}")
    if url:
        fact_parts.append(f"- **Endpoint**: {url}")
    if auth:
        fact_parts.append(f"- **Authorization**: {auth}")
    if headers_list_str:
        fact_parts.append("- **Required Headers**:")
        for h in headers_list_str:
            fact_parts.append(f"  - {h}")
    if body:
        fact_parts.append(f"- **JSON Request Body**:\n```json\n{body}\n```")
        
    if postman_steps:
        fact_parts.append("- **Postman Steps**:")
        for idx, s in enumerate(postman_steps):
            fact_parts.append(f"  {idx+1}. {s}")
    else:
        fact_parts.append("- **Postman Steps**:")
        fact_parts.append("  1. Open Postman and create a new request.")
        if method:
            fact_parts.append(f"  2. Set the HTTP method to {method}.")
        if url:
            fact_parts.append(f"  3. Enter the endpoint: {url}")
        if auth:
            fact_parts.append(f"  4. Configure Authorization as: {auth}")
        if headers_list_str:
            fact_parts.append("  5. Under Headers, configure the required headers.")
        if body:
            fact_parts.append("  6. Under Body, select raw and JSON, and enter the request body.")
        fact_parts.append("  7. Click Send.")
        
    return "\n".join(fact_parts)


def render_user_prompt(user_question: str, chunks: List[DocumentChunk], target_operation: str = "None", query_type: str = "None") -> str:
    """
    Render the final user prompt by formatting the context blocks and query.
    """
    if target_operation and target_operation != "None":
        context_text = format_fact_sheet(chunks, target_operation, query_type=query_type)
    else:
        context_text = format_context(chunks, include_metadata=True)
        
    return USER_PROMPT_TEMPLATE.format(
        context_text=context_text,
        target_operation=target_operation,
        user_question=user_question
    )


def render_image_prompt(
    user_question: str, 
    ocr_text: str, 
    vision_description: str, 
    chunks: List[DocumentChunk],
    target_operation: str = "None",
    query_type: str = "None"
) -> str:
    """
    Render the user prompt for image troubleshooting queries.
    """
    if target_operation and target_operation != "None":
        context_text = format_fact_sheet(chunks, target_operation, query_type=query_type)
    else:
        context_text = format_context(chunks, include_metadata=True)
        
    return IMAGE_USER_PROMPT_TEMPLATE.format(
        context_text=context_text,
        vision_description=vision_description,
        ocr_text=ocr_text,
        target_operation=target_operation,
        user_question=user_question or "Diagnose the error shown in the screenshot and suggest solutions."
    )
