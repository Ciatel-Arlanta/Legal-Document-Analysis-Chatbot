import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Legal Document Analysis Chatbot",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class SimpleGPUVectorStore:
    """Simple GPU-accelerated vector store using PyTorch"""
    
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.documents = []
        self.embeddings = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add_documents(self, documents):
        """Add documents and compute embeddings"""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        
        # Get embeddings and move to GPU
        embeddings_list = self.embeddings_model.embed_documents(texts)
        self.embeddings = torch.tensor(embeddings_list, device=self.device, dtype=torch.float32)
        
    def similarity_search(self, query, k=5):
        """Search for similar documents using GPU acceleration"""
        if self.embeddings is None:
            return []
            
        # Get query embedding and move to GPU
        query_embedding = self.embeddings_model.embed_query(query)
        query_tensor = torch.tensor(query_embedding, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # Compute cosine similarity on GPU
        similarities = torch.cosine_similarity(query_tensor, self.embeddings, dim=1)
        
        # Get top k results
        top_k_indices = torch.topk(similarities, min(k, len(self.documents))).indices
        
        return [self.documents[i] for i in top_k_indices.cpu().numpy()]
    
    def as_retriever(self, search_kwargs=None):
        """Return a retriever interface"""
        search_kwargs = search_kwargs or {"k": 5}
        
        class Retriever:
            def __init__(self, vectorstore, search_kwargs):
                self.vectorstore = vectorstore
                self.search_kwargs = search_kwargs
                
            def get_relevant_documents(self, query):
                return self.vectorstore.similarity_search(query, **self.search_kwargs)
        
        return Retriever(self, search_kwargs)

@st.cache_resource
def load_embeddings():
    """Load embeddings model - cached for efficiency"""
    try:
        # Use a model that works well with GPU
        return HuggingFaceEmbeddings(
            model_name="law-ai/InLegalBERT",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return HuggingFaceEmbeddings()

@st.cache_resource
def load_llm():
    """Load LLM - cached for efficiency with better configuration"""
    try:
        # Use accessible, lightweight model for legal analysis
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        # Create pipeline with optimized parameters
        qa_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,  # Increased for more detailed answers
            min_length=50,   # Ensure minimum length
            do_sample=True,  # Enable sampling for more diverse outputs
            temperature=0.3, # Slightly higher for creativity but still focused
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,  # Reduce repetition
            device=0 if torch.cuda.is_available() else -1
        )
        return HuggingFacePipeline(pipeline=qa_pipeline)
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        # Fallback to smaller model
        qa_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            min_length=50,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            device=0 if torch.cuda.is_available() else -1
        )
        return HuggingFacePipeline(pipeline=qa_pipeline)

def process_pdf(uploaded_file):
    """Process uploaded PDF and create vectorstore"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        # Enhanced text splitter with legal-specific separators
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Slightly larger chunks for better context
            chunk_overlap=200,  # More overlap to ensure continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)
        
        embeddings = load_embeddings()
        
        # Option 1: Use ChromaDB (simpler installation)
        try:
            vectorstore = Chroma.from_documents(chunks, embeddings)
            st.info("Using ChromaDB as vector store")
        except Exception as chroma_error:
            # Option 2: Use custom GPU vector store
            st.info("Using custom GPU-accelerated vector store")
            vectorstore = SimpleGPUVectorStore(embeddings)
            vectorstore.add_documents(chunks)
        
        os.unlink(tmp_file_path)
        return vectorstore, len(chunks)
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None, 0

def extract_key_points(text, max_points=5):
    """Extract key points from text with improved parsing"""
    # Clean up the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Use NLTK for better sentence tokenization
    sentences = sent_tokenize(text)
    
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s) > 15]
    
    # If we have fewer sentences than max_points, return all
    if len(sentences) <= max_points:
        return sentences
    
    # Score sentences based on length and legal terms
    legal_terms = [
        "shall", "must", "agreement", "contract", "party", "obligation", 
        "liability", "termination", "payment", "breach", "warranty", 
        "indemnify", "governing law", "dispute", "confidential"
    ]
    
    sentence_scores = []
    for sentence in sentences:
        # Base score from length (prefer medium-length sentences)
        length_score = min(len(sentence.split()) / 20, 1.0)
        
        # Legal term score
        legal_score = sum(1 for term in legal_terms if term.lower() in sentence.lower())
        legal_score = min(legal_score / 3, 1.0)  # Normalize
        
        # Position score (earlier sentences might be more important)
        position_score = 1.0 - (sentences.index(sentence) / len(sentences))
        
        # Combined score
        total_score = 0.5 * length_score + 0.3 * legal_score + 0.2 * position_score
        sentence_scores.append((sentence, total_score))
    
    # Sort by score and return top sentences
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in sentence_scores[:max_points]]

def create_conversation_chain(vectorstore):
    """Create conversation chain with enhanced legal-specific prompt"""
    llm = load_llm()
    if llm is None:
        return None
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Enhanced system prompt with legal expertise and structure
    prompt_template = """You are a legal expert with deep knowledge of contract law and legal document analysis. 

Your task is to provide precise, accurate answers to questions about the legal document provided.

ANALYSIS GUIDELINES:
1. Carefully analyze the context from the document
2. Identify the specific legal concepts relevant to the question
3. Provide a direct answer that addresses the question
4. Support your answer with specific references to the document when possible
5. Use clear, professional legal terminology
6. Structure your answer with the most important information first

CONTEXT FROM DOCUMENT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}

Provide a comprehensive, legally accurate answer to the question above:
"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),  # Increased for better context
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False
    )
    
    return conversation_chain

def analyze_document_structure(vectorstore):
    """Analyze document for common legal clauses"""
    if not vectorstore:
        return {}
    
    common_clauses = [
        "payment", "termination", "indemnity", "governing law",
        "dispute resolution", "confidentiality", "liability",
        "assignment", "force majeure", "obligations"
    ]
    
    analysis = {}
    for clause in common_clauses:
        try:
            docs = vectorstore.similarity_search(clause, k=3)
            analysis[clause] = "Found" if any(clause.lower() in doc.page_content.lower() for doc in docs) else "Not clearly found"
        except:
            analysis[clause] = "Error checking"
    
    return analysis

# Display GPU information
def show_gpu_info():
    """Display GPU information"""
    if torch.cuda.is_available():
        st.sidebar.success(f"🚀 GPU Available: {torch.cuda.get_device_name()}")
        st.sidebar.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        st.sidebar.warning("🔋 Using CPU (No GPU detected)")

# Streamlit UI
st.title("⚖️ Legal Document Analysis Chatbot")
st.markdown("Upload any legal document (PDF) and ask questions about its terms, clauses, or obligations. Get precise, clause-specific answers.")

# Sidebar for file upload and document analysis
with st.sidebar:
    show_gpu_info()
    
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a legal document (PDF)",
        type="pdf",
        help="Upload contracts, NDAs, leases, wills, or other legal documents"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing legal document..."):
                vectorstore, chunk_count = process_pdf(uploaded_file)
                
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation_chain = create_conversation_chain(vectorstore)
                    st.success(f"✅ Document processed! Created {chunk_count} text chunks.")
                    
                    st.subheader("📋 Clause Analysis")
                    analysis = analyze_document_structure(vectorstore)
                    for clause, status in analysis.items():
                        if status == "Found":
                            st.success(f"✅ {clause.title()}: {status}")
                        elif status == "Not clearly found":
                            st.warning(f"⚠️ {clause.title()}: {status}")
                        else:
                            st.error(f"❌ {clause.title()}: {status}")

# FAQ Section
st.header("❓ Frequently Asked Questions (FAQs)")
st.markdown("Select an FAQ to quickly analyze common legal document terms.")
faq_questions = [
    "What are the payment obligations in the document?",
    "What are the termination conditions?",
    "What are the parties' main obligations?",
    "Is there a governing law clause?",
    "What are the dispute resolution terms?",
    "Are there confidentiality provisions?",
    "Can rights or obligations be assigned?",
    "Is there a force majeure clause?",
    "What are the indemnity provisions?"
]
faq_selection = st.selectbox("Choose an FAQ:", ["Select an FAQ"] + faq_questions)

# Main chat interface
if st.session_state.conversation_chain is not None:
    st.header("💬 Ask Questions About Your Legal Document")
    
    if st.session_state.chat_history:
        st.markdown("### 📜 Conversation History")
        for i, (question, answer) in enumerate(reversed(st.session_state.chat_history[-5:]), 1):  # Show last 5
            with st.expander(f"💬 {question[:60]}{'...' if len(question) > 60 else ''}", expanded=False):
                st.markdown(f"**Q:** {question}")
                st.markdown(f"**A:** {answer}")
                st.markdown("---")
    
    user_question = st.text_input(
        "Ask a question about your document:",
        placeholder="e.g., What are the termination conditions? Is there a confidentiality clause?"
    )
    
    st.subheader("🔍 Quick Analysis Options")
    col1, col2, col3 = st.columns(3)
    
    quick_question = None
    with col1:
        if st.button("Summarize Document"):
            quick_question = "Provide a comprehensive summary of the legal document."
    
    with col2:
        if st.button("Key Terms"):
            quick_question = "What are the key terms and conditions in the document?"
    
    with col3:
        if st.button("Party Obligations"):
            quick_question = "What are the main obligations of each party in the document?"
    
    question_to_process = faq_selection if faq_selection != "Select an FAQ" else quick_question if quick_question else user_question
    
    if question_to_process:
        with st.spinner("Analyzing your question..."):
            try:
                response = st.session_state.conversation_chain({"question": question_to_process})
                answer = response["answer"]
                
                # Clean up the answer
                answer = answer.strip()
                
                # Format the answer better
                st.session_state.chat_history.append((question_to_process, answer))
                
                # Display answer in a nice card format
                st.markdown("### 📝 Answer")
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                    <p style="font-size: 16px; line-height: 1.6; color: #333; margin: 0;">
                        {answer}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show key points with improved parsing
                st.markdown("#### 🔑 Key Points")
                key_points = extract_key_points(answer, max_points=4)
                
                for i, point in enumerate(key_points, 1):
                    # Highlight legal terms in the point
                    highlighted_point = point
                    legal_terms = ["shall", "must", "agreement", "contract", "party", "obligation", 
                                  "liability", "termination", "payment", "breach", "warranty"]
                    
                    for term in legal_terms:
                        if term.lower() in highlighted_point.lower():
                            # Use regex to match whole words only
                            pattern = r'\b(' + re.escape(term) + r')\b'
                            highlighted_point = re.sub(
                                pattern, 
                                f'<strong style="color: #FF5722;">{term}</strong>', 
                                highlighted_point, 
                                flags=re.IGNORECASE
                            )
                    
                    st.markdown(f"""
                    <div style="background-color: black; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #2196F3;">
                        <strong style="color: #2196F3;">Point {i}:</strong> {highlighted_point}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Source references in a cleaner format
                if "source_documents" in response and response["source_documents"]:
                    with st.expander("📚 View Source References", expanded=False):
                        for i, doc in enumerate(response["source_documents"][:3], 1):
                            st.markdown(f"**Reference {i}** (Page {doc.metadata.get('page', 'N/A')})")
                            st.text_area(
                                label=f"Context {i}",
                                value=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                                height=100,
                                disabled=True,
                                key=f"source_{i}_{len(st.session_state.chat_history)}"
                            )
                            if i < len(response["source_documents"][:3]):
                                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error processing question: {e}")
                st.info("Try rephrasing your question or ask something more specific.")

else:
    st.info("👆 Please upload a legal document PDF using the sidebar to get started.")
    
    st.subheader("📝 What You Can Do:")
    st.markdown("""
    - **Upload** contracts, NDAs, leases, wills, or other legal documents
    - **Ask about** specific clauses, terms, or obligations
    - **Use FAQs** to explore common legal provisions
    - **Get summaries** or detailed clause analysis
    - **Understand** rights and obligations of parties
    """)

# Footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This tool provides informational analysis of legal documents and is not a substitute for professional legal advice. Consult a qualified lawyer for legal guidance.")