import os
import platform
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Asistente Inteligente de PDFs",
    page_icon="📚",
    layout="centered"
)

# --- ENCABEZADO PRINCIPAL ---
st.title("📚 Asistente Inteligente de PDFs")
st.caption("Versión de Python: " + platform.python_version())
st.write(
    "Esta aplicación utiliza **Generación Aumentada por Recuperación (RAG)** para responder preguntas "
    "sobre el contenido de un documento PDF. Solo necesitas subir un archivo y escribir tus preguntas."
)

# --- IMAGEN DECORATIVA ---
try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=350, caption="Tu asistente para analizar documentos PDF 📄")
except Exception:
    st.warning("⚠️ No se pudo cargar la imagen decorativa.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("ℹ️ Sobre esta aplicación")
    st.write(
        "El modelo utiliza **OpenAI GPT-4o** junto con técnicas de búsqueda semántica. "
        "Esto permite generar respuestas precisas basadas en el contenido de tu documento."
    )
    st.divider()
    st.write("🔒 Tus datos no se almacenan. Todo el análisis ocurre localmente durante la sesión.")

# --- API KEY ---
ke = st.text_input("🔑 Ingresa tu clave de OpenAI:", type="password")
if ke:
    os.environ["OPENAI_API_KEY"] = ke
else:
    st.warning("Por favor ingresa tu clave de API para continuar.")

# --- SUBIDA DE PDF ---
st.subheader("📤 Cargar documento PDF")
pdf = st.file_uploader("Selecciona un archivo PDF", type=["pdf"])

# --- PROCESAMIENTO ---
if pdf is not None and ke:
    try:
        with st.spinner("📖 Extrayendo texto del PDF..."):
            pdf_reader = PdfReader(pdf)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

        if not text.strip():
            st.error("No se pudo extraer texto del PDF. Asegúrate de que no esté escaneado como imagen.")
        else:
            st.success(f"✅ Texto extraído correctamente ({len(text)} caracteres)")

            # Dividir en fragmentos
            with st.spinner("🔍 Dividiendo el documento en fragmentos..."):
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
            st.info(f"📑 Documento dividido en {len(chunks)} fragmentos para su análisis.")

            # Crear embeddings
            with st.spinner("🧠 Creando base de conocimiento..."):
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)

            # --- INTERFAZ DE PREGUNTAS ---
            st.divider()
            st.subheader("💬 Haz una pregunta sobre el documento")
            user_question = st.text_area("Escribe tu pregunta aquí:")

            if user_question:
                with st.spinner("🤔 Buscando la mejor respuesta..."):
                    docs = knowledge_base.similarity_search(user_question)
                    llm = OpenAI(temperature=0, model_name="gpt-4o")
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_question)

                st.markdown("### 🧾 Respuesta:")
                st.success(response)

                # Mostrar contexto opcional
                with st.expander("📚 Ver fragmentos de contexto utilizados"):
                    for i, doc in enumerate(docs[:3]):
                        st.markdown(f"**Fragmento {i+1}:**")
                        st.write(doc.page_content)

    except Exception as e:
        st.error("🚨 Ocurrió un error al procesar el PDF.")
        st.code(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("⚠️ Ingresa tu clave de API antes de analizar el PDF.")
else:
    st.info("💡 Sube un archivo PDF para comenzar el análisis.")
