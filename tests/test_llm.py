import unittest
from unittest.mock import patch, Mock, MagicMock
from fastapi import HTTPException
from requests import Session
from app.crud.llm import _call_mistral, _call_openai, extract_sql_query, generate_sparql_query, generate_sql_query


class TestSQLAndSparqlGeneration(unittest.TestCase):

    def setUp(self):
        self.db_mock = MagicMock(spec=Session)  # Mock the database session
        self.sample_schema = "CREATE TABLE files (id INT, name VARCHAR(100));"
        self.sample_query = "Show all files"
    
    # Test the generate_sql_query function with OpenAI model
    @patch('app.crud.llm._call_openai')
    def test_generate_sql_query_openai(self, mock_openai):
        mock_openai.return_value = "SELECT * FROM files"
        sql_query = generate_sql_query(self.sample_query, self.db_mock, "openai")
        self.assertEqual(sql_query, "SELECT * FROM files")

    # Test the generate_sql_query function with Mistral model
    @patch('app.crud.llm._call_mistral')
    def test_generate_sql_query_mistral(self, mock_mistral):
        mock_mistral.return_value = "SELECT * FROM files"
        sql_query = generate_sql_query(self.sample_query, self.db_mock, "mistral")
        self.assertEqual(sql_query, "SELECT * FROM files")

    # Test the generate_sparql_query function with OpenAI model
    @patch('app.crud.llm._call_openai')
    @patch('app.crud.llm.generate_ttl')
    def test_generate_sparql_query_openai(self, mock_generate_ttl, mock_openai):
        mock_openai.return_value = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
        sparql_query = generate_sparql_query(self.sample_query, self.db_mock, "openai")
        self.assertEqual(sparql_query, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }")

    # Test the generate_sparql_query function with Mistral model
    @patch('app.crud.llm._call_mistral')
    @patch('app.crud.llm.generate_ttl')
    def test_generate_sparql_query_mistral(self, mock_generate_ttl, mock_mistral):
        mock_mistral.return_value = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
        sparql_query = generate_sparql_query(self.sample_query, self.db_mock, "mistral")
        self.assertEqual(sparql_query, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }")

    # Test the _call_openai function to ensure valid response is parsed correctly
    @patch('openai.ChatCompletion.create')
    def test_call_openai_valid_response(self, mock_openai):
        mock_openai.return_value = Mock(choices=[Mock(message={"content": '{"sql_query": "SELECT * FROM files"}'})])
        sql_query = _call_openai("system message", self.sample_query, "sql")
        self.assertEqual(sql_query, "SELECT * FROM files")

    # Test the _call_mistral function to ensure valid response is parsed correctly
    @patch('requests.post')
    def test_call_mistral_valid_response(self, mock_post):
        mock_post.return_value = Mock(status_code=200, json=Mock(return_value={
            "choices": [{"message": {"content": '{"sql_query": "SELECT * FROM files"}'}}]
        }))
        sql_query = _call_mistral("system message", self.sample_query, "sql")
        self.assertEqual(sql_query, "SELECT * FROM files")

    # Test extract_sql_query to ensure proper extraction from valid JSON response
    def test_extract_sql_query_valid(self):
        response_content = '{"sql_query": "SELECT * FROM files"}'
        sql_query = extract_sql_query(response_content)
        self.assertEqual(sql_query, "SELECT * FROM files")

    # Test extract_sql_query to ensure it raises an error for invalid JSON
    def test_extract_sql_query_invalid(self):
        response_content = "Invalid response"
        with self.assertRaises(HTTPException):
            extract_sql_query(response_content)

if __name__ == '__main__':
    unittest.main()
