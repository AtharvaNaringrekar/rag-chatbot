import unittest
from uuid import uuid4
from ingestion.chunkers import RecursiveCharacterChunker
from core.models import DocumentChunk


class TestRecursiveCharacterChunker(unittest.TestCase):
    """
    Unit tests for the RecursiveCharacterChunker.
    """

    def setUp(self):
        self.doc_id = uuid4()
        self.base_metadata = {"filename": "test_doc.txt", "author": "QA Team"}

    def test_chunk_text_small_fit(self):
        """
        Text smaller than chunk_size should return a single chunk.
        """
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        text = "Hello world! This is a simple test text that is quite short."
        
        chunks = chunker.chunk_text(text, self.doc_id, self.base_metadata)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, text)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].document_id, self.doc_id)
        self.assertEqual(chunks[0].metadata["filename"], "test_doc.txt")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)

    def test_chunk_text_recursive_split(self):
        """
        Text larger than chunk_size should split recursively by paragraph boundaries first.
        """
        # Set chunk_size small to force split
        chunker = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=5)
        text = "Paragraph one is here.\n\nParagraph two is there."
        
        chunks = chunker.chunk_text(text, self.doc_id, self.base_metadata)
        
        self.assertGreater(len(chunks), 1)
        # Should split by paragraph boundary first
        self.assertIn("Paragraph one", chunks[0].content)
        self.assertIn("Paragraph two", chunks[1].content)

    def test_chunk_text_overlap(self):
        """
        Consecutive chunks should overlap by the configured character size.
        """
        chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=4)
        text = "a b c d e f g h i j"
        
        chunks = chunker.chunk_text(text, self.doc_id, self.base_metadata)
        
        self.assertGreater(len(chunks), 1)
        # Verify that the end of chunk 0 has overlap elements present in chunk 1
        c0 = chunks[0].content
        c1 = chunks[1].content
        # Let's verify that the last few characters of chunk 0 are present in chunk 1
        overlap_segment = c0[-3:]  # check last 3 chars
        self.assertIn(overlap_segment, c1)

    def test_empty_text(self):
        """
        Empty text inputs should yield empty list of chunks.
        """
        chunker = RecursiveCharacterChunker()
        chunks = chunker.chunk_text("", self.doc_id, self.base_metadata)
        self.assertEqual(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
