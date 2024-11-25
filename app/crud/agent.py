import os
import warnings
from dotenv import load_dotenv
from sqlalchemy.exc import SAWarning
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase


# Ignorar advertencias de ciclos en la base de datos
warnings.filterwarnings("ignore", category=SAWarning)

load_dotenv()

db_uri = os.getenv("DATABASE_URL")
db = SQLDatabase.from_uri(db_uri)  
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
agent = create_sql_agent(llm=llm, db=db, verbose=True)


# Definir la función para obtener el esquema de la base de datos
def get_limited_schema(question):
    try:
        # Obtener la información del esquema de la base de datos
        raw_table_info = db.get_table_info()
        
        # Procesar solo si el esquema es texto
        if isinstance(raw_table_info, str):
            schema_lines = raw_table_info.splitlines()
            relevant_lines = [
                line for line in schema_lines if any(keyword in line.lower() for keyword in question.lower().split())
            ]
            return "\n".join(relevant_lines) if relevant_lines else "No se encontraron tablas relevantes."
        else:
            return "No se pudo procesar el esquema en el formato esperado."
    except Exception as e:
        print(f"Error en get_limited_schema: {e}")
        return "Error: No se pudo procesar el esquema."

# Función para reformular los resultados en lenguaje natural
def humanize_response(sql_query, sql_result):
    try:
        if sql_query.strip() == "SELECT 1;":
            return "La consulta no está diseñada para devolver datos útiles. Por favor, revisa la pregunta inicial."

        if isinstance(sql_result, list):
            sql_result_str = "\n".join(
                [", ".join([f"{key}: {value}" for key, value in row.items()]) for row in sql_result]
            )
        else:
            sql_result_str = str(sql_result)

        prompt = ChatPromptTemplate.from_template("""
        He ejecutado la siguiente consulta SQL:
        {sql_query}

        Los resultados obtenidos son los siguientes:
        {sql_result_str}

        Reformula esta información para que sea comprensible y útil en un lenguaje natural.
        """)

        humanized_response = llm.invoke(prompt.format_prompt(
            sql_query=sql_query,
            sql_result_str=sql_result_str
        ).to_messages())
        return humanized_response
    except Exception as e:
        print(f"Error en humanize_response: {e}")
        return "No se pudo reformular el resultado."


# Crear el agente SQL
agent = create_sql_agent(
    llm=llm,
    toolkit=None,
    db=db,
    verbose=True,
    handle_parsing_errors=True,
)

# Función para procesar consultas con el agente
def process_query_with_sql_agent(question):
    try:
        # Ejecuta la consulta usando el agente
        response = agent.run(question)
        print("Generated Response:", response)
        return response
    except Exception as e:
        print(f"Error en la ejecución del agente: {e}")
        return "Error al procesar la consulta."



