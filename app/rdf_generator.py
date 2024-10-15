import os
import json
import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD
import sqlalchemy
from sqlalchemy import MetaData, inspect
from dotenv import load_dotenv
from app.db.db import get_database_schema

load_dotenv()

def create_knowledge_graph():
    knowledge_graph = Graph()
    data_ns = Namespace("http://example.org/data/")
    voc_ns = Namespace("http://example.org/vocabulary/")

    knowledge_graph.bind("data", data_ns)
    knowledge_graph.bind("voc", voc_ns)
    knowledge_graph.bind("rdf", RDF)
    knowledge_graph.bind("rdfs", RDFS)

    return knowledge_graph, data_ns, voc_ns

def generate_label(row):
    return str(row.iloc[1]) if len(row) > 1 else f"Resource {row.iloc[0]}"

def add_table_to_graph(knowledge_graph, table_name, table_info, voc_ns, data_ns):
    knowledge_graph.add((voc_ns[table_name], RDF.type, RDFS.Class))
    knowledge_graph.add((voc_ns[table_name], RDFS.label, Literal(table_name, datatype=XSD.string)))

    for column in table_info['columns']:
        column_name = column['COLUMN_NAME']
        data_type = column['DATA_TYPE']
        property_uri = data_ns[column_name]
        knowledge_graph.add((property_uri, RDF.type, RDF.Property))
        knowledge_graph.add((property_uri, RDFS.label, Literal(column_name, datatype=XSD.string)))
        knowledge_graph.add((property_uri, RDFS.range, Literal(data_type, datatype=XSD.string)))

def add_rows_to_graph(knowledge_graph, table, df, foreign_keys, inspector, voc_ns, data_ns):
    for _, row in df.iterrows():
        subject = URIRef(data_ns[f"{table}/{row.iloc[0]}"])
        knowledge_graph.add((subject, RDF.type, voc_ns[table]))

        label = generate_label(row)
        knowledge_graph.add((subject, RDFS.label, Literal(label, datatype=XSD.string)))

        for column in df.columns:
            value = row[column]
            if column in [fk['constrained_columns'][0] for fk in foreign_keys]:
                ref_table = next(fk['referred_table'] for fk in foreign_keys if fk['constrained_columns'][0] == column)
                ref_subject = URIRef(data_ns[f"{ref_table}/{value}"])
                knowledge_graph.add((subject, voc_ns[f"has_{ref_table}"], ref_subject))
            else:
                add_column_value_to_graph(knowledge_graph, subject, data_ns[column], value)

def add_column_value_to_graph(knowledge_graph, subject, column_uri, value):
    if isinstance(value, str):
        knowledge_graph.add((subject, column_uri, Literal(value, datatype=XSD.string)))
    elif isinstance(value, (int, float)):
        knowledge_graph.add((subject, column_uri, Literal(value, datatype=XSD.double)))
    else:
        knowledge_graph.add((subject, column_uri, Literal(str(value), datatype=XSD.string)))

def generate_ttl(output_path="output.ttl"):
    db_url = os.getenv('DATABASE_URL')
    engine = sqlalchemy.create_engine(db_url)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    inspector = inspect(engine)

    knowledge_graph, data_ns, voc_ns = create_knowledge_graph()

    schema_dict = json.loads(get_database_schema())

    for table_name, table_info in schema_dict.items():
        add_table_to_graph(knowledge_graph, table_name, table_info, voc_ns, data_ns)

    for table in metadata.tables.keys():
        df = pd.read_sql_table(table, engine)
        foreign_keys = inspector.get_foreign_keys(table)
        add_rows_to_graph(knowledge_graph, table, df, foreign_keys, inspector, voc_ns, data_ns)

    knowledge_graph.serialize(destination=output_path, format="turtle")
    return output_path

def generate_schema(output_path="schema_output.ttl"):
    db_url = os.getenv('DATABASE_URL')
    engine = sqlalchemy.create_engine(db_url)

    # Inicializar metadata e inspector
    metadata = MetaData()
    metadata.reflect(bind=engine)
    inspector = inspect(engine)

    # Crear el grafo RDF usando la función existente
    knowledge_graph, data_ns, voc_ns = create_knowledge_graph()

    # Agregar tipo de grafo RDF
    knowledge_graph.add((voc_ns['DatabaseSchema'], RDF.type, RDFS.Class))
    knowledge_graph.add((voc_ns['DatabaseSchema'], RDFS.label, Literal("Database Schema", datatype=XSD.string)))

    for table in metadata.tables.keys():
        table_uri = voc_ns[table]
        # Añadir cada tabla al grafo
        knowledge_graph.add((table_uri, RDF.type, RDFS.Class))
        knowledge_graph.add((table_uri, RDFS.label, Literal(table, datatype=XSD.string)))

        # Obtener claves foráneas
        foreign_keys = inspector.get_foreign_keys(table)

        for column in metadata.tables[table].columns:
            column_uri = voc_ns[f"{table}/{column.name}"]
            # Añadir cada columna al grafo
            knowledge_graph.add((column_uri, RDF.type, RDF.Property))
            knowledge_graph.add((column_uri, RDFS.label, Literal(column.name, datatype=XSD.string)))

            # Agregar el rango de cada columna
            knowledge_graph.add((column_uri, RDFS.range, Literal(str(column.type), datatype=XSD.string)))

            # Agregar relaciones si la columna es una clave foránea
            if column.name in [fk['constrained_columns'][0] for fk in foreign_keys]:
                ref_table = next(fk['referred_table'] for fk in foreign_keys if fk['constrained_columns'][0] == column.name)
                ref_table_uri = voc_ns[ref_table]
                knowledge_graph.add((table_uri, voc_ns[f"has_{ref_table}"], ref_table_uri))

    # Serializar el grafo RDF a un archivo
    knowledge_graph.serialize(destination=output_path, format="turtle")
    return output_path