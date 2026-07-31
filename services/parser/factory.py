import os
import json
from typing import Dict, Any
from core.interfaces.parser import IDocumentParser
from core.exceptions import UnsupportedFormatException
from services.parser.doc_parsers import PDFParser, DocxParser, TxtParser
from services.parser.openapi import OpenAPIParser
from services.parser.postman import PostmanCollectionParser


class DocumentParserFactory:
    """
    Factory class to resolve the correct IDocumentParser implementation
    based on the file extension and JSON structural sniffing.
    """

    @staticmethod
    def get_parser(filename: str, file_content: bytes) -> IDocumentParser:
        """
        Determine the appropriate parser based on extension and contents.

        Args:
            filename: Name of the uploaded file.
            file_content: Raw bytes of the file (used for JSON format sniffing).

        Returns:
            An instance of IDocumentParser.

        Raises:
            UnsupportedFormatException: If the file type cannot be parsed.
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext == ".pdf":
            return PDFParser()
        elif ext == ".docx":
            return DocxParser()
        elif ext == ".txt":
            return TxtParser()
        elif ext in (".yaml", ".yml"):
            # OpenAPI specifications are commonly YAML
            return OpenAPIParser()
        elif ext == ".json":
            # Sniff JSON contents to distinguish OpenAPI from Postman
            return DocumentParserFactory._sniff_json(filename, file_content)
        else:
            raise UnsupportedFormatException(ext)

    @staticmethod
    def _sniff_json(filename: str, file_content: bytes) -> IDocumentParser:
        """
        Sniff JSON keys to detect Postman Collections vs OpenAPI Swagger definitions.
        """
        try:
            content_str = file_content.decode("utf-8", errors="ignore")
            data = json.loads(content_str)
            
            if not isinstance(data, dict):
                # Fallback to general TXT parsing or throw? Let's treat as text
                return TxtParser()

            # OpenAPI identifiers
            if "openapi" in data or "swagger" in data or "paths" in data:
                return OpenAPIParser()
            
            # Postman identifiers
            info = data.get("info", {})
            if isinstance(info, dict) and ("schema" in info or "postman_id" in info or "_postman_id" in info):
                return PostmanCollectionParser()
            if "item" in data and isinstance(data["item"], list):
                # Another strong Postman indicator
                return PostmanCollectionParser()
            
            # Default fallback for random JSON: treat as OpenAPI if it has paths,
            # otherwise treat as a text dump using TXT parser
            return TxtParser()

        except Exception:
            # If JSON decoding fails here, let the OpenAPI parser handle it and raise a ParsingException
            return OpenAPIParser()
