import streamlit as st
import requests
import os
from streamlit_toggle import st_toggle_switch
from dotenv import load_dotenv
load_dotenv()

sparql_endpoint = os.getenv("SPARQL_ENDPOINT")
sql_endpoint = os.getenv("SQL_ENDPOINT")

st.markdown("<h1 style='text-align: center;'>Charla con la Base de Datos</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Centro Movil</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .bubble-user {
        background-color: #cce7ff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: left;
        color: black;
        width: fit-content;
        margin-left: auto; 
    }
    .bubble-database {
        background-color: #ffccf2;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: left;
        color: black;
        width: fit-content;
    }
    .model-change {
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if "input" not in st.session_state:
    st.session_state.input = ""

if "model" not in st.session_state:
    st.session_state.model = False

if "use_knowledge_graph" not in st.session_state:
    st.session_state.use_knowledge_graph = False

for chat in st.session_state.history:
    if 'model_change' in chat:
        st.markdown(f"<div class='model-change'>{chat['model_change']}</div>", unsafe_allow_html=True)
    elif 'knowledge_graph_change' in chat:
        st.markdown(f"<div class='model-change'>{chat['knowledge_graph_change']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bubble-user'>{chat['user']}</div>", unsafe_allow_html=True) 
        st.markdown(f"<div class='bubble-database'>{chat['response']}</div>", unsafe_allow_html=True)

def send_message():
    user_input = st.session_state.input
    if user_input:
        model_name = "openai" if st.session_state.model else "mistral"
        endpoint = sparql_endpoint if st.session_state.use_knowledge_graph else sql_endpoint        
        payload = {
            "user_query": user_input,
            "model": model_name
        }
        try:
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                data = response.json()
                if "results" in data and isinstance(data["results"], str):
                    results = data["results"]
                else:
                
                    results = "No results found "
            else:
                    results = f"Error: {response.text}"

        except Exception as e:
            results = f"Error: {str(e)}"

        st.session_state.history.append({"user": user_input, "response": results})
        st.session_state.input = ""

def set_input_text(message):
    st.session_state.input = message

def toggle_model(new_model):
    st.session_state.model = new_model 
    new_model_name = "OpenAI" if st.session_state.model else "Mistral"
    st.session_state.history.append({"model_change": f"Se cambio al modelo {new_model_name}"})
    st.rerun()

def toggle_knowledge_graph(new_use_kg):
    st.session_state.use_knowledge_graph = new_use_kg
    new_state_message = "Usando grafo de conocimiento" if new_use_kg else "Usando Base de Datos SQL"
    st.session_state.history.append({"knowledge_graph_change": f" {new_state_message}"})
    st.rerun()

st.write("Tú:")
user_input = st.text_input("", key="input", on_change=send_message, placeholder="Escribe tu mensaje aquí...", label_visibility="collapsed")

common_messages = [
    "Ver para el cliente (nombre del cliente) cantidad de ordenes por estado",
    "Ver para el cliente (nombre del cliente) la orden con el precio mas alto",
    "Ver para el cliente (nombre del cliente) la cantidad de ordenes en total",
    "Soy el vendedor (nombre del vendedor), lista los nombres de mis clientes por ciudad"
]

st.sidebar.markdown("### Mensajes frecuentes")
for msg in common_messages:
        st.sidebar.button(msg, on_click=set_input_text, args=(msg,))

col1, col2 = st.columns([1, 2])

with col1:
    knowledge_graph_status = "Usando: Grafo Conocimiento" if st.session_state.use_knowledge_graph else "Usando: SQL"
    st.markdown(f"<p style='text-align: left; font-weight: bold;'>{knowledge_graph_status}</p>", unsafe_allow_html=True)    
    kg_switch = st_toggle_switch(
        label="Knowledge Graph",  
        key="knowledge_graph_toggle",
        default_value=st.session_state.use_knowledge_graph,  
        label_after="Deactivate Knowledge Graph",
        inactive_color="#D3D3D3", 
        active_color="#11567f",
        track_color="#29B5E8",
    )
    if kg_switch != st.session_state.use_knowledge_graph:
        toggle_knowledge_graph(kg_switch)

with col2:
    model_status = "Usando: OpenAI" if st.session_state.model else "Usando: Mistral"
    st.markdown(f"<p style='text-align: right; font-weight: bold;'>{model_status}</p>", unsafe_allow_html=True)    
    model_switch = st_toggle_switch(
        label="OpenAI",  
        key="toggle",
        default_value=st.session_state.model,  
        label_after="Deactivate OpenAI",
        inactive_color="#D3D3D3", 
        active_color="#11567f",
        track_color="#29B5E8",
    )
    if model_switch != st.session_state.model:
        toggle_model(model_switch)

st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)