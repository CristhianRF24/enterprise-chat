import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from rdflib import Graph

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4")

kg = Graph()
kg.parse("output.ttl", format="turtle")  

def get_knowledge_graph_schema():
    schema = set()
    for s, p, o in kg.triples((None, None, None)):
        schema.add((s.n3(), p.n3(), o.n3()))
    return "\n".join([f"{s} {p} {o}" for s, p, o in schema])

def execute_sparql_query(query):
    try:
        result = kg.query(query)
        formatted_result = [
            {str(var): str(binding[var]) for var in binding}
            for binding in result
        ]
        return formatted_result if formatted_result else "No se encontraron resultados."
    except Exception as e:
        return f"Error ejecutando SPARQL: {e}"

tools = [
    Tool(
        name="Get Knowledge Graph Schema",
        func=lambda _: get_knowledge_graph_schema(),
        description="Obtiene el esquema del Knowledge Graph."
    ),
    Tool(
        name="Execute SPARQL Query",
        func=lambda inputs: execute_sparql_query(inputs["query"]),
        description="Ejecuta una consulta SPARQL en el Knowledge Graph y devuelve los resultados."
    ),
]

prompt = ChatPromptTemplate.from_template(
    template=(
        "Eres un asistente que utiliza un Knowledge Graph para responder preguntas. "
        "Usa las herramientas disponibles para consultar información relevante. "
        "Si necesitas conocer la estructura del Knowledge Graph, usa 'Get Knowledge Graph Schema'. "
        "Si necesitas ejecutar una consulta SPARQL, usa 'Execute SPARQL Query'. "
        "Asegúrate de devolver respuestas claras y basadas en el Knowledge Graph."
    )
)


# prompt = ChatPromptTemplate.from_template(
#  template=(
#         "Debes responder las preguntas del usuario mediante el knowledge graph. "
#     ))

# # Crear el agente
# agent = create_react_agent(
#     tools=tools,
#     llm=llm, 
#     prompt=prompt  
# )

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description", 
    verbose=True  
)

def process_query_with_kg_agent(question):
    try:
        response = agent.invoke({"input": question})
        return response
    except Exception as e:
        return f"Error procesando la consulta: {e}"

if __name__ == "__main__":
    question = (
        "Dame los datos de todas las personas y sus mascotas."
    )
    result = process_query_with_kg_agent(question)
    print("Respuesta final:", result)