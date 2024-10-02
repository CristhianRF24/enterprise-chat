
import os
import openai
from requests import Session
from app.api.v1.endpoints.files import get_database_schema



def generate_sql_query(user_query: str, db: Session):
   #get database schema
    schema = get_database_schema(db)
    
    system_message = f"""
    Given the following schema, write a SQL query that retrieves the requested information.
    Return the SQL query inside a JSON structure with the key "sql_query".
    <example>
    {{
        "sql_query": "SELECT * FROM files;"
        "original_query": "Show me all the files."
    }}
    </example>
    <schema>
    {schema}
    </schema>
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ]
    )
    
    return response.choices[0].message.content