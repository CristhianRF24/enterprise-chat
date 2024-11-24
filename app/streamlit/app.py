import streamlit as st
import requests
import os
from streamlit_toggle import st_toggle_switch
from dotenv import load_dotenv

load_dotenv()

sparql_endpoint = os.getenv("SPARQL_ENDPOINT")
sql_endpoint = os.getenv("SQL_ENDPOINT")
pdf_upload_endpoint = os.getenv("PDF_UPLOAD_ENDPOINT")
ask_model_endpoint = os.getenv("ASK_AGENT_ENDPOINT")

if "page" not in st.session_state:
    st.session_state.page = "chat_bdd"

def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()

def chat_page():
    st.markdown("<h1 style='text-align: center;'>Charla con la universidad</h1>", unsafe_allow_html=True)

    if st.button("Ir a Chatear con PDFs"):
        navigate_to("chat_pdf")

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
        "Hola, soy (nombre del cliente). ¿Podrían decirme cuántas órdenes tengo actualmente en cada estado?",
        "Hola, quiero ver cuál es la orden con el precio más alto para el cliente (nombre del cliente) . ¿Me podrían ayudar?",
        "Hola, quiero ver para el cliente (nombre del cliente) las ordenes por estado",
        "Hola, soy el vendedor (nombre del vendedor) y quiero ver todos los nombres de mis clientes por ciudad",
        "Hola, por favor muestrame (n) reclamos que tenga en el estado 'EN PROGRESO'",
        "Hola, por favor muestrame (n) reclamos por estado",
        "Hola, por favor muestrame (n) reclamos que tenga estado 'RESUELTO'",
        "Hola, muestrame (n) reclamos del cliente (nombre del cliente) que tenga estado 'EN PROGRESO'"
        "Hola, soy el cliente (nombre del cliente) y quiero ver mis (n) ultimos pagos realizados"
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

def chat_pdf():
    st.markdown("<h1 style='text-align: center;'>Chat con PDF</h1>", unsafe_allow_html=True)
    
    # Botón para navegar a otra página si es necesario
    if st.button("Ir a Chatear con BDD"):
        navigate_to("chat_bdd")
    
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
        </style>
        """, unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    if "input" not in st.session_state:
        st.session_state.input = ""

    for chat in st.session_state.history:
        st.markdown(f"<div class='bubble-user'>{chat['user']}</div>", unsafe_allow_html=True) 
        st.markdown(f"<div class='bubble-database'>{chat['response']}</div>", unsafe_allow_html=True)

    def send_message():
        user_input = st.session_state.input
        if user_input:
            payload = {'query': user_input, 'model': "string"}
            try:
                response = requests.post(ask_model_endpoint, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    output = data["response"].get("output", "No se obtuvo salida del modelo.")
                    if output == "Agent stopped due to iteration limit or time limit.":
                        results = "El modelo se detuvo debido a los límites de tiempo o iteración."
                    else:
                        results = output
                else:
                        print("aqui dentro")
                        results = f"Error: {response.text}"

            except Exception as e:
                print("aqui fuera")
                results = f"Error: {str(e)}"

            st.session_state.history.append({"user": user_input, "response": results})
            st.session_state.input = ""

    st.write("Tú:")
    st.text_input("", key="input", on_change=send_message, placeholder="Escribe tu mensaje aquí...", label_visibility="collapsed")

    # Botón para subir PDF
    uploaded_pdf = st.file_uploader("Subir archivo PDF", type=["pdf"])
    
    if uploaded_pdf is not None:
        st.write("Archivo PDF cargado. Iniciando carga al servidor...")
        files = {'file': uploaded_pdf}
        try:
            response = requests.post(pdf_upload_endpoint, files=files)
            if response.status_code == 200:
                st.success("PDF cargado con éxito.")
            else:
                st.error(f"Error al cargar el archivo: {response.text}")
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")

    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

# Renderiza la página actual
if st.session_state.page == "chat_bdd":
    chat_page()
elif st.session_state.page == "chat_pdf":
    chat_pdf()