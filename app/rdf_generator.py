import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD
import sqlalchemy
from sqlalchemy import MetaData, inspect
import json
from app.db.db import get_database_schema

def generate_ttl(output_path="output.ttl"):
    db_url = "mysql+pymysql://root@localhost:3306/enterprice_chat"
    engine = sqlalchemy.create_engine(db_url)

    metadata = MetaData()
    metadata.reflect(bind=engine)

    inspector = inspect(engine)
    g = Graph()

    data_ns = Namespace("http://example.org/data/")
    voc_ns = Namespace("http://example.org/vocabulary/")
    g.bind("data", data_ns)
    g.bind("voc", voc_ns)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)

    # Cambiar la función de generación de etiqueta
    def generate_label(row):
        # Usa la segunda columna (índice 1) para crear la etiqueta
        if len(row) > 1:  # Verifica que haya al menos dos columnas
            return str(row.iloc[1])  # Usa el valor de la segunda columna como etiqueta
        return f"Resource {row.iloc[0]}"

    schema = get_database_schema()
    schema_dict = json.loads(schema)

    for table_name, table_info in schema_dict.items():
        g.add((voc_ns[table_name], RDF.type, RDFS.Class))
        g.add((voc_ns[table_name], RDFS.label, Literal(table_name, datatype=XSD.string)))

        for column in table_info['columns']:
            column_name = column['COLUMN_NAME']
            data_type = column['DATA_TYPE']
            property_uri = data_ns[column_name]
            g.add((property_uri, RDF.type, RDF.Property))
            g.add((property_uri, RDFS.label, Literal(column_name, datatype=XSD.string)))
            g.add((property_uri, RDFS.range, Literal(data_type, datatype=XSD.string)))

    for table in metadata.tables.keys():
        df = pd.read_sql_table(table, engine)
        foreign_keys = inspector.get_foreign_keys(table)

        for _, row in df.iterrows():
            subject = URIRef(data_ns[f"{table}/{row.iloc[0]}"])
            g.add((subject, RDF.type, voc_ns[table]))

            # Usa la función modificada para generar la etiqueta
            label = generate_label(row)
            g.add((subject, RDFS.label, Literal(label, datatype=XSD.string)))

            for column in df.columns:
                value = row[column]
                if column in [fk['constrained_columns'][0] for fk in foreign_keys]:
                    ref_table = [fk['referred_table'] for fk in foreign_keys if fk['constrained_columns'][0] == column][0]
                    ref_subject = URIRef(data_ns[f"{ref_table}/{value}"])
                    g.add((subject, voc_ns[f"has_{ref_table}"], ref_subject))
                else:
                    if isinstance(value, str):
                        g.add((subject, data_ns[column], Literal(value, datatype=XSD.string)))
                    elif isinstance(value, (int, float)):
                        g.add((subject, data_ns[column], Literal(value, datatype=XSD.double)))
                    else:
                        g.add((subject, data_ns[column], Literal(str(value), datatype=XSD.string)))

    g.serialize(destination=output_path, format="turtle")
    return output_path