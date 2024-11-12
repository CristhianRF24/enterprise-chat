# ENTERPRISE CHAT

## Description
This project is a business chat bot aimed at improving customer service.
## Índice
- [Installation](#installation)
- [Use](#use)
## Instalación
1. Clone this repository.
   ```bash
   git clone https://github.com/CristhianRF24/enterprise-chat.git
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Create a .env file in the root of the project with the following data
    ```bash
    OPENAI_API_KEY=YOUR_KEY
    API_URL=https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions
    HUGGINGFACE_TOKEN=hf_YOUR_TOKEN

    DATABASE_URL=mysql+mysqlconnector://root:PASSWORD@localhost:3306/YOUR_BD_NAME
    DB_HOST=localhost
    DB_USER=root
    DB_PASS=PASSWORD
    DB_NAME=YOUR_BD_NAME
    DB_PORT=3306
    GRAPHDB_URL = http://localhost:7200
    REPO_NAME = knowledge_graph
    SPARQL_ENDPOINT=http://127.0.0.1:8000/generate_sparql/
    SQL_ENDPOINT=http://127.0.0.1:8000/generate_sql/
    ```

## Use
1. Run the following command to start the backend project:
   ```bash
   uvicorn app.main:app
   ```
2. To launch the project interface, open another terminal and navigate to the appropriate folder:
   ```bash
   cd .\app\streamlit\
   ```
3. Then, run the following command to start the interface with Streamlit:
   ```bash
   streamlit run app.py  
   ```

4. (Optional) Set up the Knowledge Graph in GraphDB for visualization
   To visualize the knowledge graph interactively, you can install and configure GraphDB.

   - Download and install GraphDB following the instructions on their website.
   - Start GraphDB and access the interface in your browser at http://localhost:7200.
   - Create a new repository in GraphDB named knowledge_graph to upload your RDF graph.
   - After starting with step 1, you can run the backend method generate_and_load_ttl_endpoint to automatically load the database into graphdb.
Note: Ensure the GRAPHDB_URL and REPO_NAME values in your .env file match your GraphDB configuration.

