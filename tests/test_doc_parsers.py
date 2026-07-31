import unittest
from unittest.mock import patch, MagicMock
from services.parser.factory import DocumentParserFactory
from services.parser.doc_parsers import PDFParser, DocxParser, TxtParser
from core.exceptions import UnsupportedFormatException
from core.models import Document, DocumentChunk


class TestDocParsersAndFactory(unittest.TestCase):
    """
    Unit tests for PDFParser, DocxParser, TxtParser, and DocumentParserFactory.
    """

    def test_factory_resolves_standard_types(self):
        """
        DocumentParserFactory should instantiate correct parsers for known extensions.
        """
        pdf_parser = DocumentParserFactory.get_parser("test.pdf", b"")
        self.assertIsInstance(pdf_parser, PDFParser)

        docx_parser = DocumentParserFactory.get_parser("test.docx", b"")
        self.assertIsInstance(docx_parser, DocxParser)

        txt_parser = DocumentParserFactory.get_parser("test.txt", b"")
        self.assertIsInstance(txt_parser, TxtParser)

        yaml_parser = DocumentParserFactory.get_parser("test.yaml", b"")
        from services.parser.openapi import OpenAPIParser
        self.assertIsInstance(yaml_parser, OpenAPIParser)

    def test_factory_raises_unsupported(self):
        """
        DocumentParserFactory should raise UnsupportedFormatException for unrecognized files.
        """
        with self.assertRaises(UnsupportedFormatException):
            DocumentParserFactory.get_parser("test.exe", b"")

    @patch("services.parser.doc_parsers.PdfReader")
    def test_pdf_parser_page_by_page(self, mock_pdf_reader):
        """
        PDFParser should iterate page-by-page and record page number metadata.
        """
        # Setup mock reader and pages
        mock_reader_inst = MagicMock()
        mock_pdf_reader.return_value = mock_reader_inst
        
        page_1 = MagicMock()
        page_1.extract_text.return_value = "Page one content."
        page_2 = MagicMock()
        page_2.extract_text.return_value = "Page two content."
        
        mock_reader_inst.pages = [page_1, page_2]

        parser = PDFParser()
        doc, chunks = parser.parse(b"dummy_pdf_bytes", "test.pdf")

        self.assertEqual(doc.filename, "test.pdf")
        self.assertEqual(doc.file_type, "pdf")
        self.assertEqual(doc.metadata["total_pages"], 2)
        
        # Verify page number assignments in chunks
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].content, "Page one content.")
        self.assertEqual(chunks[0].metadata["page_number"], 1)
        self.assertEqual(chunks[1].content, "Page two content.")
        self.assertEqual(chunks[1].metadata["page_number"], 2)

    @patch("services.parser.doc_parsers.DocxDocument")
    def test_docx_parser_heading_and_tables(self, mock_docx):
        """
        DocxParser should parse paragraphs, detect active headings, and parse tables.
        """
        mock_doc_inst = MagicMock()
        mock_docx.return_value = mock_doc_inst

        # Paragraphs
        p1 = MagicMock()
        p1.text = "Document Title"
        p1.style.name = "Title"

        p2 = MagicMock()
        p2.text = "This is introductory text."
        p2.style.name = "Normal"

        p3 = MagicMock()
        p3.text = "API Methods"
        p3.style.name = "Heading 1"

        p4 = MagicMock()
        p4.text = "Use this endpoint to authenticate."
        p4.style.name = "Normal"

        mock_doc_inst.paragraphs = [p1, p2, p3, p4]

        # Mock tables (empty for simplicity, or 1 mock table)
        mock_doc_inst.tables = []

        parser = DocxParser()
        doc, chunks = parser.parse(b"dummy_docx_bytes", "test.docx")

        self.assertEqual(doc.filename, "test.docx")
        self.assertEqual(doc.file_type, "docx")
        
        self.assertGreater(len(chunks), 0)
        # First chunk should have section "Document Title" or "Introduction"
        self.assertEqual(chunks[0].metadata["section"], "Document Title")
        # Second chunk should have section "API Methods"
        self.assertEqual(chunks[1].metadata["section"], "API Methods")
        self.assertIn("endpoint to authenticate", chunks[1].content)

    def test_txt_parser_simple(self):
        """
        TxtParser should decode byte content and return standard chunks.
        """
        parser = TxtParser()
        text_content = "Hello Technical Support Bot! This is plain text."
        doc, chunks = parser.parse(text_content.encode("utf-8"), "test.txt")

        self.assertEqual(doc.filename, "test.txt")
        self.assertEqual(doc.file_type, "txt")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, text_content)
        self.assertEqual(chunks[0].metadata["filename"], "test.txt")


if __name__ == "__main__":
    unittest.main()
