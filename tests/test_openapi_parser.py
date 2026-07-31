import unittest
from services.parser.openapi import OpenAPIParser
from core.models import DocumentChunk


class TestOpenAPIParser(unittest.TestCase):
    """
    Unit tests for the semantic OpenAPIParser.
    """

    def setUp(self):
        self.parser = OpenAPIParser()
        
        # Define a mock JSON OpenAPI spec
        self.openapi_json = """{
            "openapi": "3.0.0",
            "info": {
                "title": "Store API Gateway",
                "version": "2.1.0",
                "description": "Core storefront services"
            },
            "servers": [{"url": "https://gateway.store.com/api"}],
            "paths": {
                "/orders": {
                    "post": {
                        "summary": "Create purchase order",
                        "operationId": "createOrder",
                        "parameters": [
                            {
                                "name": "X-Client-Id",
                                "in": "header",
                                "required": true,
                                "schema": {"type": "string"}
                            }
                        ],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/OrderPayload"
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "Order created successfully"
                            }
                        }
                    }
                }
            },
            "components": {
                "schemas": {
                    "OrderPayload": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "quantity": {"type": "integer"}
                        }
                    }
                }
            }
        }"""

    def test_parse_json_spec_correctly(self):
        """
        OpenAPIParser should correctly ingest JSON strings, resolve references, and format endpoint chunks.
        """
        doc, chunks = self.parser.parse(self.openapi_json.encode("utf-8"), "openapi.json")

        # Verify Document Metadata
        self.assertEqual(doc.filename, "openapi.json")
        self.assertEqual(doc.file_type, "openapi")
        self.assertEqual(doc.metadata["title"], "Store API Gateway")
        self.assertEqual(doc.metadata["version"], "2.1.0")
        self.assertEqual(doc.metadata["base_url"], "https://gateway.store.com/api")

        # Expect chunks:
        # 1. Global metadata chunk
        # 2. POST /orders endpoint chunk
        # 3. OrderPayload schema component chunk
        self.assertEqual(len(chunks), 3)

        # Global Metadata check
        self.assertEqual(chunks[0].metadata["type"], "global_metadata")

        # Endpoint Chunk check
        endpoint_chunk = chunks[1]
        self.assertEqual(endpoint_chunk.metadata["type"], "endpoint")
        self.assertEqual(endpoint_chunk.metadata["api_path"], "/orders")
        self.assertEqual(endpoint_chunk.metadata["http_method"], "POST")
        self.assertEqual(endpoint_chunk.metadata["operation_id"], "createOrder")

        # Ensure Markdown content contains our parameters and schemas
        self.assertIn("## Endpoint: POST /orders", endpoint_chunk.content)
        self.assertIn("X-Client-Id", endpoint_chunk.content)
        self.assertIn("item_id", endpoint_chunk.content)      # references resolved inline
        self.assertIn("quantity", endpoint_chunk.content)     # references resolved inline

        # Schema Chunk check
        schema_chunk = chunks[2]
        self.assertEqual(schema_chunk.metadata["type"], "schema")
        self.assertEqual(schema_chunk.metadata["schema_name"], "OrderPayload")
        self.assertIn("OrderPayload", schema_chunk.content)

    def test_parse_yaml_spec_correctly(self):
        """
        OpenAPIParser should also parse standard YAML syntax definitions.
        """
        openapi_yaml = """
openapi: 3.0.0
info:
  title: Mini API
  version: 1.0.0
paths:
  /ping:
    get:
      summary: Health check
      responses:
        '200':
          description: OK
"""
        doc, chunks = self.parser.parse(openapi_yaml.encode("utf-8"), "spec.yaml")
        
        self.assertEqual(doc.metadata["title"], "Mini API")
        self.assertEqual(len(chunks), 2)  # global info + /ping endpoint
        self.assertEqual(chunks[1].metadata["api_path"], "/ping")
        self.assertEqual(chunks[1].metadata["http_method"], "GET")


if __name__ == "__main__":
    unittest.main()
