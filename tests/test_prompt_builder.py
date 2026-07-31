import unittest
from uuid import uuid4
from core.models import DocumentChunk
from pipeline.prompts import format_context, render_user_prompt, SYSTEM_PROMPT


class TestPromptBuilder(unittest.TestCase):
    """
    Unit tests for context formatting and prompt rendering inside pipeline/prompts.py.
    """

    def setUp(self):
        self.doc_id = uuid4()

    def test_format_context_empty(self):
        """
        If no chunks are provided, format_context should return a fallback string.
        """
        formatted = format_context([])
        self.assertEqual(formatted, "No documentation context available.")

    def test_format_context_with_metadata(self):
        """
        format_context should parse and compile metadata fields (api_path, page, section) correctly.
        """
        chunk1 = DocumentChunk(
            document_id=self.doc_id,
            content="Get products returns 200.",
            chunk_index=0,
            metadata={
                "filename": "api_spec.json",
                "api_path": "/products",
                "http_method": "GET"
            }
        )
        chunk2 = DocumentChunk(
            document_id=self.doc_id,
            content="Check configuration values.",
            chunk_index=1,
            metadata={
                "filename": "manual.pdf",
                "page_number": 4,
                "section": "Setup"
            }
        )

        formatted = format_context([chunk1, chunk2])

        # Assert format includes descriptors
        self.assertIn("--- Context Block 1 [Source: api_spec.json, Endpoint: GET /products] ---", formatted)
        self.assertIn("Get products returns 200.", formatted)
        self.assertIn("--- Context Block 2 [Source: manual.pdf, Page: 4, Section: Setup] ---", formatted)
        self.assertIn("Check configuration values.", formatted)

    def test_render_user_prompt(self):
        """
        render_user_prompt should combine formatted context and the question correctly.
        """
        chunk = DocumentChunk(
            document_id=self.doc_id,
            content="Retrieve token.",
            chunk_index=0,
            metadata={"filename": "auth.txt"}
        )
        
        prompt = render_user_prompt(
            user_question="How to log in?",
            chunks=[chunk]
        )

        self.assertIn("### TECHNICAL DOCUMENTATION CONTEXT:", prompt)
        self.assertIn("Retrieve token.", prompt)
        self.assertIn("### USER QUESTION:\nHow to log in?", prompt)

    def test_system_prompt_constraints(self):
        """
        Verify the system prompt contains the strict fallback phrase constraint.
        """
        self.assertIn(
            "I couldn't find that information in the uploaded documentation.",
            SYSTEM_PROMPT
        )
        self.assertIn("Answer the user query using ONLY", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
