from typing import List
from core.models import DocumentChunk

# Strict System Prompt persona and constraints
# Strict System Prompt prefix (rules and constraints)
SYSTEM_PROMPT_PREFIX = (
    "You are a Spintly API Support Engineer. Answer the user query using ONLY the provided documentation context. "
    "Be direct, concise, and professional. Do NOT speak like a search engine or mention documents.\n\n"
    "CRITICAL RULES & CPU LATENCY GUARDRAILS:\n"
    "1. General Fallback: If information is missing, respond EXACTLY: \"I could not find this information in the available Spintly documentation.\" "
    "(Ensure the exact fallback phrase \"I couldn't find that information in the uploaded documentation.\" is also matched.)\n"
    "2. Troubleshooting Fallback: If troubleshooting steps are missing, respond EXACTLY: \"The retrieved Spintly documentation does not provide specific troubleshooting steps for this issue.\"\n"
    "3. Zero Hallucinations: Do not invent endpoints, parameters, or request bodies. Base answers strictly on the retrieved context. Never mix context with pre-trained LLM knowledge or output AI cutoff/Microsoft disclaimers.\n"
    "4. Hide Credentials: Replace API tokens/secrets with placeholders like <ACCESS_TOKEN>.\n"
    "5. Hide Metadata: Do NOT display document names, page numbers, sections, or context block IDs. Never use phrases like 'according to the context' or 'the document states'. Speak natively.\n"
    "6. Format: Use structured markdown (code blocks, headers, lists). Omit empty sections or placeholders entirely.\n"
    "7. Conciseness: Keep answers under 3 sentences (max 100 words) unless step-by-step instructions are requested. Do NOT generate marketing/generic filler text (e.g. 'enhancing operational efficiency') or use bullet points in explanations.\n"
    "8. Heading rule: Headings must be short and exactly as template defined; do not put explanations inside headings.\n\n"
    "RESPONSE TEMPLATE & ADAPTIVE STRUCTURES:\n"
    "Before generating, classify the user query into the target category and format your response using ONLY the sections specified in that category template:\n\n"
)

TEMPLATE_DEFINITION = (
    "DEFINITION / CONCEPT QUESTIONS (e.g., 'What is Spintly Cloud?', 'Explain OAuth'):\n"
    "Format your response as a single, direct, concise paragraph (3-5 lines, max 80 words) explaining the concept. "
    "Do NOT include headings like '## Question Summary', '## Answer', or '## Key Points'."
)

TEMPLATE_SHORT_FACT = (
    "SHORT FACT QUESTIONS (e.g., 'Which API lists all users?', 'What is the OAuth endpoint?'):\n"
    "Format your response as a single, direct, concise sentence (max 30 words) answering the question. "
    "Do NOT include headings like '## Answer' or bullet points."
)

TEMPLATE_HOW_TO = (
    "HOW-TO / PROCEDURE QUESTIONS (e.g., 'How do I create a user?', 'How do I obtain an OAuth access token?'):\n"
    "## Step-by-Step Instructions\n"
    "[Numbered list of instructions to complete the task]\n\n"
    "## Example Request\n"
    "[Markdown code block of the request, ONLY if available in documentation]"
)

TEMPLATE_API_ENDPOINT = (
    "API ENDPOINT QUESTIONS (e.g., 'Explain Create User API', 'What does Get Sites API do?'):\n"
    "## API Purpose\n"
    "[Brief description of what the endpoint does on the line below the heading]\n\n"
    "## Endpoint Details\n"
    "- **HTTP Method**: [Method in uppercase, e.g. POST, GET]\n"
    "- **URL**: [URL endpoint path, e.g. /saamsIdm/oauth/token]\n"
    "- **Authentication**: [Authentication method required, or None]\n\n"
    "## Request Body\n"
    "[If the documentation contains a request body, render it as properly formatted JSON inside a markdown code block (json)]\n\n"
    "## Success Response\n"
    "[Success status code and body details, if available]\n\n"
    "## Example Request\n"
    "[Markdown code block of example call, ONLY if available]"
)

TEMPLATE_TROUBLESHOOTING = (
    "TROUBLESHOOTING QUESTIONS (e.g., 'Why am I getting 401?', 'User cannot access the door'):\n"
    "## Problem Cause\n"
    "[Concise explanation of the cause of the problem on the line below the heading]\n\n"
    "## Resolution Steps\n"
    "[Numbered list of troubleshooting steps. Every step must be supported by the retrieved documentation. If no supporting documentation exists, state that troubleshooting steps are not available.]"
)

TEMPLATE_COMPARISON = (
    "COMPARISON QUESTIONS (ONLY when the user query explicitly asks to compare two items, e.g., 'Type 1 vs Type 2 Integration', 'Site vs Access Point'):\n"
    "## Overview\n"
    "[Short introduction paragraph]\n\n"
    "## Comparison Details\n"
    "| Feature/Aspect | [Concept 1] | [Concept 2] |\n"
    "| :--- | :--- | :--- |\n"
    "| [Comparison Row 1] | [Value 1] | [Value 2] |\n\n"
    "## Key Differences\n"
    "[Bulleted list of key comparison differences]\n"
)

# Full System Prompt assembled for back-compatibility and testing checks
SYSTEM_PROMPT = (
    SYSTEM_PROMPT_PREFIX +
    "1. " + TEMPLATE_DEFINITION + "\n\n" +
    "2. " + TEMPLATE_SHORT_FACT + "\n\n" +
    "3. " + TEMPLATE_HOW_TO + "\n\n" +
    "4. " + TEMPLATE_API_ENDPOINT + "\n\n" +
    "5. " + TEMPLATE_TROUBLESHOOTING + "\n\n" +
    "6. " + TEMPLATE_COMPARISON
)


# User Prompt Template
USER_PROMPT_TEMPLATE = (
    "### TECHNICAL DOCUMENTATION CONTEXT:\n"
    "{context_text}\n\n"
    "### USER QUESTION:\n"
    "{user_question}\n\n"
    "### SUPPORT ENGINEER RESPONSE:"
)


def format_context(chunks: List[DocumentChunk], include_metadata: bool = True) -> str:
    """
    Format a list of retrieved DocumentChunks into a clean, structured context block
    that the LLM can easily parse.
    """
    if not chunks:
        return "No documentation context available."
        
    context_blocks = []
    for idx, chunk in enumerate(chunks):
        if include_metadata:
            filename = chunk.metadata.get("filename", "unknown_file")
            desc_parts = [f"Source: {filename}"]
            
            endpoint_name = chunk.metadata.get("endpoint_name")
            category = chunk.metadata.get("category")
            http_method = chunk.metadata.get("http_method")
            url_path = chunk.metadata.get("url") or chunk.metadata.get("api_path")
            auth_req = chunk.metadata.get("authentication")
            
            if endpoint_name:
                desc_parts.append(f"API Endpoint: {endpoint_name}")
            if http_method and url_path:
                desc_parts.append(f"Endpoint: {http_method} {url_path}")
            if category:
                desc_parts.append(f"Category: {category}")
            if auth_req:
                desc_parts.append(f"Auth: {auth_req}")
            if "page_number" in chunk.metadata:
                desc_parts.append(f"Page: {chunk.metadata['page_number']}")
            if "section" in chunk.metadata:
                desc_parts.append(f"Section: {chunk.metadata['section']}")
                
            header = f"--- Context Block {idx + 1} [{', '.join(desc_parts)}] ---"
        else:
            header = f"--- Context Block {idx + 1} ---"
            
        block = f"{header}\n{chunk.content}"
        context_blocks.append(block)
        
    return "\n\n".join(context_blocks)


def render_user_prompt(user_question: str, chunks: List[DocumentChunk]) -> str:
    """
    Render the final user prompt by formatting the context blocks and query.
    """
    context_text = format_context(chunks, include_metadata=False)
    return USER_PROMPT_TEMPLATE.format(
        context_text=context_text,
        user_question=user_question
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
    "### USER QUESTION:\n"
    "{user_question}\n\n"
    "### SUPPORT ENGINEER RESPONSE:"
)


def render_image_prompt(
    user_question: str, 
    ocr_text: str, 
    vision_description: str, 
    chunks: List[DocumentChunk]
) -> str:
    """
    Render the user prompt for image troubleshooting queries.
    """
    context_text = format_context(chunks, include_metadata=False)
    return IMAGE_USER_PROMPT_TEMPLATE.format(
        context_text=context_text,
        vision_description=vision_description,
        ocr_text=ocr_text,
        user_question=user_question or "Diagnose the error shown in the screenshot and suggest solutions."
    )
