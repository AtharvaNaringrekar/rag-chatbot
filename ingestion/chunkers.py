from typing import List, Dict, Any
from uuid import UUID
from core.models import DocumentChunk
from config.settings import settings


class RecursiveCharacterChunker:
    """
    Splits text into smaller, overlapping chunks by recursively analyzing
    a hierarchy of separators. This ensures paragraph, line, and word boundaries
    are preserved as much as possible.
    """

    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
        separators: List[str] = None
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum character length of a single chunk.
            chunk_overlap: Character overlap between consecutive chunks.
            separators: Separator characters to try, ordered from highest priority (paragraphs)
                        to lowest priority (individual characters).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Default separators: paragraphs, newlines, spaces, empty string
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk_text(self, text: str, document_id: UUID, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split a block of text into a list of DocumentChunks.

        Args:
            text: The normalized text content of the document.
            document_id: The UUID of the parent document.
            base_metadata: Dictionary of parent document metadata to inherit (e.g. filename).

        Returns:
            A list of DocumentChunk domain models.
        """
        if not text:
            return []

        # Recursively split the text string
        raw_chunks = self._split_text(text, self.separators)
        
        # Merge small split pieces to reach target chunk_size with chunk_overlap
        merged_texts = self._merge_splits(raw_chunks)

        # Map merged text strings into DocumentChunk models
        chunks = []
        for index, chunk_content in enumerate(merged_texts):
            # Create a shallow copy of base metadata to avoid reference sharing
            chunk_metadata = base_metadata.copy()
            # Mark the approximate page or section if present in the text (often parsed by PDF/DOCX)
            # Add index metadata
            chunk_metadata["chunk_index"] = index
            
            chunk = DocumentChunk(
                document_id=document_id,
                content=chunk_content,
                chunk_index=index,
                metadata=chunk_metadata
            )
            chunks.append(chunk)

        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the list of separators.
        """
        # If the text is already under the target size, no more splitting needed
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # No separators left; force slice the text into chunk_size chunks
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # Select the highest priority separator
        separator = separators[0]
        splits = []

        if separator == "":
            # Handle empty string separator (character splitting)
            splits = list(text)
        else:
            # Split using regex or standard split; we escape regex characters except space/newlines
            splits = text.split(separator)

        # Process each split piece
        final_splits = []
        for split in splits:
            if not split.strip() and separator != "":
                continue
            
            # Re-add separator to maintain formatting if it's a newline or paragraph
            if separator in ("\n\n", "\n") and separator != splits[-1]:
                item = split + separator
            else:
                item = split

            if len(item) <= self.chunk_size:
                final_splits.append(item)
            else:
                # If still too large, split recursively with the next separator
                sub_splits = self._split_text(item, separators[1:])
                final_splits.extend(sub_splits)

        return final_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge individual splits into chunks of target size, maintaining the configured overlap.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split)
            
            # If adding this split exceeds chunk_size, save current chunk and start a new one
            if current_length + split_len > self.chunk_size and current_chunk:
                merged_chunk = "".join(current_chunk)
                chunks.append(merged_chunk)
                
                # Backtrack to maintain overlap
                # We pop splits from the end of the current chunk list until we satisfy the overlap
                overlap_chunk = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(split)
            current_length += split_len

        # Append any remaining text
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks


import re
from utils.api_metadata_extractor import extract_api_metadata_from_text


class SemanticAPIChunker:
    """
    Semantic chunker designed specifically for API documentation.
    Splits text by logical API sections (e.g. Markdown headers, HTTP routes)
    and attaches rich structured API metadata to each chunk.
    """

    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def chunk_text(self, text: str, document_id: UUID, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        if not text:
            return []

        # 1. Split text into logical API sections using headers and endpoints as indicators
        # Splits on Markdown headers (H1 to H4) or lines like "POST /api/v1/users"
        split_pattern = r'(^|\n)(?=(?:#{1,4}\s+|[A-Z]{3,6}\s+/(?:api|oauth|v\d+)/))'
        raw_sections = re.split(split_pattern, text)
        
        # Filter empty sections and merge them back with their separators if split created fragments
        sections = []
        current_sec = ""
        for sec in raw_sections:
            if not sec:
                continue
            if sec.startswith("\n") and len(sec.strip()) == 0:
                current_sec += sec
            elif sec.strip().startswith("#") or re.match(r'^[A-Z]{3,6}\s+/', sec.strip()):
                if current_sec.strip():
                    sections.append(current_sec)
                current_sec = sec
            else:
                current_sec += sec
                
        if current_sec.strip():
            sections.append(current_sec)

        # Fallback: if splitting resulted in no sections, treat the entire text as one section
        if not sections:
            sections = [text]

        chunks = []
        for sec_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
                
            # Compute section-specific API metadata
            api_meta = extract_api_metadata_from_text(section)
            
            # If the logical section fits in a single chunk, keep it together
            if len(section) <= self.chunk_size:
                chunk_meta = base_metadata.copy()
                chunk_meta.update(api_meta)
                chunk_meta["logical_section_index"] = sec_idx
                
                chunk = DocumentChunk(
                    document_id=document_id,
                    content=section,
                    chunk_index=len(chunks),
                    metadata=chunk_meta
                )
                chunks.append(chunk)
            else:
                # If too large, split recursively but keep the same API metadata on all sub-chunks!
                sub_chunks = self.recursive_chunker.chunk_text(
                    text=section,
                    document_id=document_id,
                    base_metadata=base_metadata
                )
                for sub in sub_chunks:
                    # Update each sub-chunk with rich API metadata
                    sub.metadata.update(api_meta)
                    sub.metadata["logical_section_index"] = sec_idx
                    sub.chunk_index = len(chunks)
                    chunks.append(sub)

        return chunks
