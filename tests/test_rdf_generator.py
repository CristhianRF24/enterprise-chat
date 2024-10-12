import json
import unittest
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from app.rdf_generator import add_rows_to_graph, add_table_to_graph, create_knowledge_graph, generate_ttl

class TestGenerateTTL(unittest.TestCase):

    #Set up a mock environment for testing.
    def setUp(self):
        self.mock_engine = patch('sqlalchemy.create_engine').start()
        self.mock_conn = MagicMock()
        self.mock_engine.return_value = self.mock_conn

    #Tear down mock environment.
    def tearDown(self):
        patch.stopall()

    # Test the creation of the RDF knowledge graph and namespace bindings.
    def test_create_knowledge_graph(self):
        knowledge_graph, data_ns, voc_ns = create_knowledge_graph()
        self.assertIsInstance(knowledge_graph, Graph)
        self.assertEqual(data_ns["example"], URIRef("http://example.org/data/example"))
        self.assertEqual(voc_ns["example"], URIRef("http://example.org/vocabulary/example"))

    # Test that a table and its columns are added to the RDF graph correctly.
    @patch('app.db.db.get_database_schema')
    def test_add_table_to_graph(self, mock_get_database_schema):
        mock_get_database_schema.return_value = '{"test_table": {"columns": [{"COLUMN_NAME": "id", "DATA_TYPE": "integer"}, {"COLUMN_NAME": "name", "DATA_TYPE": "string"}]}}'
        
        knowledge_graph, data_ns, voc_ns = create_knowledge_graph()
        
        schema = json.loads(mock_get_database_schema())
        
        table_info = schema["test_table"]
        add_table_to_graph(knowledge_graph, "test_table", table_info, voc_ns, data_ns)

        self.assertIn((voc_ns["test_table"], RDF.type, RDFS.Class), knowledge_graph)
        self.assertIn((data_ns["id"], RDF.type, RDF.Property), knowledge_graph)
        self.assertIn((data_ns["name"], RDF.type, RDF.Property), knowledge_graph)

    # Test that rows of data from a table are added to the RDF graph correctly.
    @patch('pandas.read_sql_table')
    def test_add_rows_to_graph(self, mock_read_sql_table):
        knowledge_graph, data_ns, voc_ns = create_knowledge_graph()
        mock_read_sql_table.return_value = pd.DataFrame({
            "id": [1, 2],
            "name": ["Test1", "Test2"]
        })

        add_rows_to_graph(knowledge_graph, "test_table", mock_read_sql_table(), [], self.mock_conn, voc_ns, data_ns)

        subject_1 = URIRef(data_ns["test_table/1"])
        subject_2 = URIRef(data_ns["test_table/2"])

        self.assertIn((subject_1, RDF.type, voc_ns["test_table"]), knowledge_graph)
        self.assertIn((subject_1, RDFS.label, Literal("Test1", datatype=XSD.string)), knowledge_graph)
        self.assertIn((subject_2, RDF.type, voc_ns["test_table"]), knowledge_graph)
        self.assertIn((subject_2, RDFS.label, Literal("Test2", datatype=XSD.string)), knowledge_graph)

    # Test that the generate_ttl function creates the correct TTL output file.
    @patch('sqlalchemy.create_engine')
    @patch('pandas.read_sql_table')
    @patch('app.db.db.get_database_schema')
    def test_generate_ttl(self, mock_get_database_schema, mock_read_sql_table, mock_create_engine):
        mock_get_database_schema.return_value = '{"test_table": {"columns": [{"COLUMN_NAME": "id", "DATA_TYPE": "integer"}, {"COLUMN_NAME": "name", "DATA_TYPE": "string"}]}}'
        mock_read_sql_table.return_value = pd.DataFrame({
            "id": [1, 2],
            "name": ["Test1", "Test2"]
        })

        output_path = "test_output.ttl"
        result = generate_ttl(output_path)

        self.assertTrue(os.path.exists(result))

        knowledge_graph = Graph()
        knowledge_graph.parse(result, format="turtle")
        self.assertIn((None, RDF.type, RDFS.Class), knowledge_graph)

if __name__ == '__main__':
    unittest.main()
