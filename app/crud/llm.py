
from fastapi import HTTPException
import openai
from requests import Session
from app.api.v1.endpoints.files import get_database_schema
import json



def generate_sql_query(user_query: str, db: Session) -> str:
    schema = get_database_schema(db)
    
    system_message = f"""
    Given the following schema, write a SQL query that retrieves the requested information.
    Return the SQL query inside a JSON structure with the key "sql_query".
    <example>
    {{
        "sql_query": "SELECT * FROM files",
        "original_query": "Show me all the files."
    }}
    
    </example>
    <schema>
    {schema}
    </schema>
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ]
    )

    response_content = response.choices[0].message["content"]
    
    try:
        response_json = json.loads(response_content)  
        sql_query = response_json.get("sql_query") 
        return sql_query
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the AI response.")