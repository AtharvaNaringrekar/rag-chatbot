import unittest
from services.parser.postman import PostmanCollectionParser
from core.models import DocumentChunk


class TestPostmanCollectionParser(unittest.TestCase):
    """
    Unit tests for the PostmanCollectionParser.
    """

    def setUp(self):
        self.parser = PostmanCollectionParser()
        
        # Mock Postman Collection JSON payload
        self.postman_json = """{
            "info": {
                "name": "Payment API Gateway",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
                "description": "Integration tests for processing payments"
            },
            "item": [
                {
                    "name": "Billing Folder",
                    "item": [
                        {
                            "name": "Execute Charge",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Authorization",
                                        "value": "Bearer {{token}}"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": "{\\"amount\\": 1500, \\"currency\\": \\"USD\\"}"
                                },
                                "url": {
                                    "raw": "https://api.payments.com/v1/charge",
                                    "host": ["api", "payments", "com"],
                                    "path": ["v1", "charge"],
                                    "query": [
                                        {
                                            "key": "notify",
                                            "value": "true"
                                        }
                                    ]
                                },
                                "description": "Trigger charge event"
                            },
                            "response": [
                                {
                                    "name": "Charge Successful Example",
                                    "code": 200,
                                    "status": "OK",
                                    "body": "{\\"transaction_id\\": \\"tx_987654\\"}"
                                }
                            ]
                        }
                    ]
                }
            ]
        }"""

    def test_parse_postman_collection(self):
        """
        PostmanCollectionParser should correctly parse requests, folders, headers, and responses.
        """
        doc, chunks = self.parser.parse(self.postman_json.encode("utf-8"), "collection.json")

        # Document metadata assertions
        self.assertEqual(doc.filename, "collection.json")
        self.assertEqual(doc.file_type, "postman")
        self.assertEqual(doc.metadata["collection_name"], "Payment API Gateway")

        # Expect chunks:
        # 1. Global metadata chunk
        # 2. POST /v1/charge request chunk
        self.assertEqual(len(chunks), 2)

        # Global Metadata check
        self.assertEqual(chunks[0].metadata["type"], "global_metadata")

        # Request Chunk check
        request_chunk = chunks[1]
        self.assertEqual(request_chunk.metadata["type"], "postman_request")
        self.assertEqual(request_chunk.metadata["api_path"], "/v1/charge")
        self.assertEqual(request_chunk.metadata["http_method"], "POST")
        self.assertEqual(request_chunk.metadata["request_name"], "Execute Charge")
        self.assertEqual(request_chunk.metadata["folder_pathway"], ["Billing Folder"])

        # Content assertions
        self.assertIn("## Request: Execute Charge", request_chunk.content)
        self.assertIn("**Folder Pathway**: `Billing Folder`", request_chunk.content)
        self.assertIn("Authorization", request_chunk.content)
        self.assertIn("transaction_id", request_chunk.content)
        self.assertIn("tx_987654", request_chunk.content)


if __name__ == "__main__":
    unittest.main()
