# --- Dependencies ---
import streamlit as st
from dotenv import load_dotenv
import os
import asyncio
import io
from PIL import Image
import fitz  
import shutil
import base64

# LangChain imports (v0.2+ style)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.generativeai import GenerativeModel

# --- Async event loop for Streamlit/Windows ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Environment ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- Custom CSS for modern UI ---
st.markdown("""
<style>

/* Tell browser we support both themes but override Streamlit defaults */
:root {
    color-scheme: light dark;
}

/* --- Main App Background --- */
body, .main, [data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #75bfec 0%, #ffffff 100%) !important;
    color: #000000 !important;
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
}

/* --- Header --- */
[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.4) !important;
}

/* --- Sidebar --- */
[data-testid="stSidebar"], .block-container {
    background: rgba(255,255,255,0.95) !important;
    color: #000000 !important;
}

/* --- Chat Message Styling --- */
.message {
    border-radius: 0.7em;
    margin-bottom: 1em;
    max-width: 84vw;
    padding: 0.8em 1.3em;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.12);
    backdrop-filter: blur(6.5px);
    font-size: 1rem;
    font-weight: 400;
}

/* User Messages - Blue */
.user-msg {
    background: #4daae3 !important;
    color: #ffffff !important;
    margin-left: auto;
    text-align: right;
}

/* Assistant Messages - White */
.assistant-msg {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #75bfec;
    margin-right: auto;
    text-align: left;
}

/* --- Card Style --- */
.card {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 1em;
    box-shadow: 0 4px 40px 0 rgba(117, 191, 236, 0.25);
    padding: 2em;
    margin-bottom: 1.3em;
    color: #000000 !important;
}

/* --- File Uploader --- */
[data-testid="stFileUploader"] {
        background: rgba(117, 191, 236, 0.1) !important;
        padding: 14px;
        border-radius: 16px;
        border: 1.5px solid #75bfec;
        color: #000000 !important;
}
[data-testid="stFileUploader"] section div div {
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 10px !important;
        border: 1px solid #75bfec !important;
}

[data-testid="stFileUploader"] section div div button {
        background: #4daae3 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold;
        padding: 6px 12px;
}
[data-testid="stFileUploader"] section div div button:hover {
    background: #3a94cc !important;
}

/* --- Buttons --- */
.stButton>button, button, [data-testid="baseButton-secondary"] {
    background: linear-gradient(90deg, #75bfec 0%, #4daae3 100%) !important;
    color: white !important;
    padding: 0.5em 1.3em;
    border-radius: 0.7em;
    font-weight: bold;
    border: none;
    letter-spacing: 1px;
}
.stButton>button:hover, button:hover {
    background: #4daae3 !important;
    color: white !important;
}

/* --- Headings --- */
h1, h2, h3, h4, h5 {
    color: #000000 !important;
    letter-spacing: 1.4px;
}

/* --- Inputs --- */
input, textarea {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #75bfec !important;
    border-radius: 0.8em !important;
}

/* --- Fix Try These Buttons --- */
a, .stMarkdown p a {
    color: #4daae3 !important;
}

/* --- Scrollbar --- */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background-color: rgba(128,128,128,0.4);
    border-radius: 4px;
}

</style>
""", unsafe_allow_html=True)



# --- Text Extraction with Gemini ---
def extract_text_from_file_gemini(uploaded_file):
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    # Try image first
    try:
        image = Image.open(io.BytesIO(file_bytes))
        model = GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(["Extract the text from this image:", image])
        return response.text
    except Exception as e:
        # Fallback to PDF parsing using PyMuPDF
        if filename.endswith(".pdf"):
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                pdf_text = ""
                for page in doc:
                    pdf_text += page.get_text()
                return pdf_text
            except Exception as pdf_e:
                return f"Error during PDF text extraction: {pdf_e}"
        else:
            return f"Unsupported file type or error: {e}"

# --- Split large text into chunks ---
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=9500, chunk_overlap=700)
    return splitter.split_text(text)

# --- Store chunks in FAISS ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# --- Set up Gemini Flash chain using Runnable (modern way) ---
def get_conversational_chain():
    prompt_template = """
    You are a helpful chatbot that answers questions about a resume.
    You are provided with the resume text as context.
    Answer the question as truthfully as possible based only on the provided context.
    If the answer is not in the provided context, just say "The resume doesn't contain this information.", don't try to make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"])
    chain = prompt | model
    return chain

# --- Chat message display with bubbles ---
def show_chat_message(message, is_user):
    bubble_class = "user-msg" if is_user else "assistant-msg"
    st.markdown(f'<div class="message {bubble_class}">{message}</div>', unsafe_allow_html=True)

# --- User question handler ---
def user_input(user_question):
    if "faiss_index" not in os.listdir():
        st.warning("Please upload and process your resume first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()

    # Show user query as bubble
    show_chat_message(user_question, is_user=True)
    with st.spinner("AI is thinking..."):
        response = chain.invoke({"context": context, "question": user_question})

    # Show model answer as bubble
    show_chat_message(response.content, is_user=False)


# --- New: Resume Improvement Suggestions using Gemini ---
def get_resume_improvement_suggestions(resume_text):
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    prompt_template = """
You are an expert resume coach.

Here is the resume text:

{resume_text}

Please provide clear, constructive, and actionable suggestions to improve this resume.
Focus on content clarity, action verbs, formatting consistency, skills emphasis, and common mistakes.

Give your feedback in bullet points.
"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | model
    response = chain.invoke({"resume_text": resume_text})
    return response.content


# --- Main UI function ---
def main():
    image_path = "speak.png"

    # Convert image to base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    # Display image + text
    st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <img src="data:image/png;base64,{image_base64}" width="40" style="border-radius:5px;">
            <h1 style="margin:0; color:black;">Resume Chatbot</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card" style="margin-top:-24px;margin-bottom:0.5em; color:black;">
            💡 <b>Ask smart questions about your resume!</b> 
            <ul style="color:black;">
              <li>Summarize skills or achievements</li>
              <li>Provide feedback on experience sections</li>
              <li>Score resume for job-readiness</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


    # 🌐 Sidebar - File upload, show preview
    with st.sidebar:
        st.image("images.png", width=80)
        st.header("Upload Resume PDF")

        # --- Reset on first load / refresh ---
        if "pdf_doc" not in st.session_state:
            st.session_state.pdf_doc = None
            st.session_state.resume_uploaded = False
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")

        # --- File uploader ---
        pdf_doc = st.file_uploader("Drag & drop or browse your PDF", type="pdf")

        # --- Detect file removal ---
        if st.session_state.pdf_doc and not pdf_doc:
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            st.session_state.pdf_doc = None
            st.session_state.resume_uploaded = False
            st.rerun()

        # --- Process PDF ---
        if st.button("Process PDF"):
            if pdf_doc:
                with st.spinner("Processing your PDF..."):
                    raw_text = extract_text_from_file_gemini(pdf_doc)
                    if raw_text and len(raw_text) > 10:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Resume processed successfully, you can now ask anything!")
                        st.session_state.pdf_doc = pdf_doc
                        st.session_state.resume_uploaded = True
                        st.session_state.raw_text = raw_text  # Cache raw text for improvements
                    else:
                        st.error("Text extraction failed. Try another file.")
            else:
                st.warning("Please upload a file.")

        # Show document preview if uploaded, else placeholder
        if st.session_state.get("resume_uploaded", False):
            st.markdown('<div style="margin:10px 0;color:#90caf9;font-size:0.9em;">✅ <b>PDF Processed!</b></div>', unsafe_allow_html=True)
            st.markdown(f'<small style="color:#ddd">Filename: <code>{st.session_state.pdf_doc.name}</code></small>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="margin-top:20px;color:#999">No resume uploaded yet.</div>', unsafe_allow_html=True)

        # --- New Button: Resume Improvement Suggestions ---
        if st.session_state.get("resume_uploaded", False):
            if st.button("Get Resume Improvement Suggestions"):
                with st.spinner("Analyzing resume for improvements..."):
                    suggestions = get_resume_improvement_suggestions(st.session_state.raw_text)
                # Show suggestions in main panel
                st.markdown(
                    f'<div class="message assistant-msg">{suggestions.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)

    # 📝 Chat interface in main panel
    st.markdown('', unsafe_allow_html=True)
    with st.form("question_form"):
        user_question = st.text_input(
            "Type a question about your resume (eg: What skills do I have? Best projects? etc)",
            key="user_query")
        submitted = st.form_submit_button("Send 👋")
        if submitted:
            if user_question.strip():
                user_input(user_question.strip())
            else:
                st.warning("Please type a question first.")

    # --- Optional: Suggested quick questions ---
    st.markdown(
        """
        <div style="margin:1.3em 0 1em 0">
        <span style="color:#01c5de;font-size:1em;">🔎 Try these:</span>
            <button onclick="window.forms['question_form'].setValue('user_query','Summarize my skills.')" style="background:#294fa7;border:none;color:#eaf7ff;border-radius:8px;padding:6px 12px;margin-left:8px;margin-right:4px;cursor:pointer;">Summarize skills</button>
            <button onclick="window.forms['question_form'].setValue('user_query','What awards are listed?')" style="background:#294fa7;border:none;color:#eaf7ff;border-radius:8px;padding:6px 12px;cursor:pointer;">Show awards</button>
            <button onclick="window.forms['question_form'].setValue('user_query','Give me feedback on my experience section.')" style="background:#294fa7;border:none;color:#eaf7ff;border-radius:8px;padding:6px 12px;cursor:pointer;">Experience feedback</button>
        </div>
        """, unsafe_allow_html=True
    )

    # Cleanup option -- for dev/testing/re-upload
    if st.button("Clear chat & resume"):
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        st.session_state.clear()
        st.rerun()


if __name__ == "__main__":
    main()



