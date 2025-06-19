import streamlit as st
import tempfile
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from transformers import pipeline
from typing import List, Dict
import pytesseract
from PIL import Image
import fitz
import io

st.set_page_config(page_title="Legal Document Analysis Chatbot", page_icon="⚖️", layout="wide")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class SimpleGPUVectorStore:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.documents = []
        self.embeddings = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add_documents(self, documents):
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        embeddings_list = self.embeddings_model.embed_documents(texts)
        self.embeddings = torch.tensor(embeddings_list, device=self.device, dtype=torch.float32)
    
    def similarity_search(self, query, k=5):
        if self.embeddings is None:
            return []
        query_embedding = self.embeddings_model.embed_query(query)
        query_tensor = torch.tensor(query_embedding, device=self.device, dtype=torch.float32).unsqueeze(0)
        similarities = torch.cosine_similarity(query_tensor, self.embeddings)
        top_k_indices = torch.topk(similarities, min(k, len(self.documents))).indices
        return [self.documents[i] for i in top_k_indices.cpu().numpy()]
    
    def as_retriever(self, search_kwargs=None):
        search_kwargs = search_kwargs or {"k": 5}
        class Retriever(BaseRetriever):
            vectorstore: SimpleGPUVectorStore
            search_kwargs: dict
            def _get_relevant_documents(self, query, *, run_manager=None):
                return self.vectorstore.similarity_search(query, **self.search_kwargs)
            async def _aget_relevant_documents(self, query, *, run_manager=None):
                return self.vectorstore.similarity_search(query, **self.search_kwargs)
        return Retriever(vectorstore=self, search_kwargs=search_kwargs)

@st.cache_resource
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="neurolab/inlegalbert", model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    except:
        st.warning("Failed to load InLegalBERT, using MiniLM.")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})

@st.cache_resource
def load_llm():
    try:
        return HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="google/flan-t5-large", max_length=512, do_sample=True, temperature=0.3, device=0 if torch.cuda.is_available() else -1))
    except:
        return HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, do_sample=True, temperature=0.3, device=0 if torch.cuda.is_available() else -1))

@st.cache_resource
def load_question_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_length=200, temperature=0.7)

def extract_text_with_ocr(pdf_path):
    """Extract text from PDF using OCR for scanned documents"""
    try:
        pdf_document = fitz.open(pdf_path)
        documents = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            text = page.get_text()
            
            if len(text.strip()) < 50:
                # Convert page to image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # Apply OCR
                ocr_text = pytesseract.image_to_string(image, config='--psm 6')
                text = ocr_text if len(ocr_text.strip()) > len(text.strip()) else text
            
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"page": page_num + 1, "source": pdf_path}
                )
                documents.append(doc)
        
        pdf_document.close()
        return documents
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
        return []

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        total_text = "".join([page.page_content for page in pages])
        if len(total_text.strip()) < 100:
            st.info("Document appears to be scanned. Using OCR...")
            pages = extract_text_with_ocr(tmp_file_path)
    except Exception as e:
        st.warning(f"Regular PDF loading failed: {e}. Trying OCR...")
        pages = extract_text_with_ocr(tmp_file_path)
    
    if not pages:
        st.error("Failed to extract text from PDF")
        os.unlink(tmp_file_path)
        return None, 0
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""])
    chunks = text_splitter.split_documents(pages)
    embeddings = load_embeddings()
    try:
        vectorstore = Chroma.from_documents(chunks, embeddings)
    except:
        vectorstore = SimpleGPUVectorStore(embeddings)
        vectorstore.add_documents(chunks)
    os.unlink(tmp_file_path)
    return vectorstore, len(chunks)

def create_conversation_chain(vectorstore):
    llm = load_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    prompt_template = """You are a legal expert analyzing contracts, NDAs, leases, or wills. Provide accurate, concise answers citing specific clauses or sections. Avoid speculation and state if information is missing. Use a professional tone.

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), memory=memory, return_source_documents=True, combine_docs_chain_kwargs={"prompt": prompt})

def generate_legal_questions(summary: str, question_generator) -> List[str]:
    prompt = f"Based on this legal document summary, generate 3 key questions about legal issues, parties, or terms:\nSummary: {summary}\nQuestions:"
    result = question_generator(prompt, num_return_sequences=1)[0]['generated_text']
    questions = [q.strip('- ').strip() for q in result.split('\n') if '?' in q and len(q.strip()) > 10]
    return questions[:3] or ["What are the main obligations of the parties?", "What are the termination conditions?", "Is there a governing law clause?"]

def analyze_document_structure(vectorstore):
    clauses = ["payment", "termination", "indemnity", "governing law", "dispute resolution", "confidentiality", "liability", "assignment", "force majeure"]
    return {clause: "Found" if any(clause.lower() in doc.page_content.lower() for doc in vectorstore.similarity_search(clause, k=3)) else "Not found" for clause in clauses}

def show_gpu_info():
    if torch.cuda.is_available():
        st.sidebar.success(f"GPU: {torch.cuda.get_device_name()}")
        st.sidebar.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        st.sidebar.warning("Using CPU")

st.title("⚖️ Legal Document Analysis Chatbot")
st.markdown("Upload a legal document (PDF) and ask questions about its terms or clauses.")

with st.sidebar:
    show_gpu_info()
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing..."):
            result = process_pdf(uploaded_file)
            if result[0] is not None:
                vectorstore, chunk_count = result
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation_chain = create_conversation_chain(vectorstore)
                st.success(f"Processed {chunk_count} chunks.")
                st.subheader("Clause Analysis")
                for clause, status in analyze_document_structure(vectorstore).items():
                    st.write(f"{'' if status == 'Found' else '⚠️'} {clause.title()}: {status}")

st.header("FAQs & Questions")
if st.session_state.vectorstore:
    summary = "".join([doc.page_content[:200] for doc in st.session_state.vectorstore.similarity_search("summary", k=3)])
    faq_questions = generate_legal_questions(summary, load_question_generator()) + [
        "What are the payment obligations?", "What are the termination conditions?", "Is there a governing law clause?"
    ]
    faq_selection = st.selectbox("Choose an FAQ:", ["Select an FAQ"] + faq_questions)

if st.session_state.conversation_chain:
    st.header("Ask Questions")
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {q[:50]}..."):
                st.write(f"**Question:** {q}\n**Answer:** {a}")
    user_question = st.text_input("Ask about the document:", placeholder="e.g., What are the confidentiality terms?")
    col1, col2, col3 = st.columns(3)
    quick_question = None
    with col1:
        if st.button("Summarize"): quick_question = "Summarize the document."
    with col2:
        if st.button("Key Terms"): quick_question = "What are the key terms?"
    with col3:
        if st.button("Obligations"): quick_question = "What are the parties' obligations?"
    question = faq_selection if faq_selection != "Select an FAQ" else quick_question or user_question
    if question:
        with st.spinner("Analyzing..."):
            try:
                response = st.session_state.conversation_chain({"question": question})
                st.session_state.chat_history.append((question, response["answer"]))
                st.subheader("Answer:")
                st.write(response["answer"])
                if response["source_documents"]:
                    with st.expander("📚 References"):
                        for i, doc in enumerate(response["source_documents"][:3]):
                            st.write(f"**Ref {i+1} (Page {doc.metadata.get('page', 'N/A')}):** {doc.page_content[:200]}...")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Upload a PDF to start.")
    st.subheader("What You Can Do:")
    st.markdown("- Upload contracts, NDAs, leases, or wills\n- Ask about clauses or terms\n- Use FAQs for common provisions\n- Get summaries or clause analysis")
st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This tool is for informational purposes only. Consult a lawyer for legal advice.")