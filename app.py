import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader, CSVLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Annoy, FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import time
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize session state variables
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = {}

# Initialize data storage lists
if 'stored_data' not in st.session_state:
    st.session_state.stored_data = []

def store_data(config_settings, retrieved_nodes, performance_metrics, comments):
    # Create a dictionary to store this run's data
    data_entry = {
        "config_settings": config_settings,
        "retrieved_nodes": retrieved_nodes,
        "performance_metrics": performance_metrics,
        "comments": comments
    }
    # Append the data entry to the stored_data list
    st.session_state.stored_data.append(data_entry)

def vector(pdf_docs, chunk_size, chunk_overlap):
    documents = []
    for csv_file in pdf_docs:
        print(f"Processing file: {csv_file.name}")
        with open(os.path.join("/tmp", csv_file.name), "wb") as f:
            f.write(csv_file.getbuffer())
        loader = CSVLoader(os.path.join("/tmp", csv_file.name))
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check and create Annoy index if not exists
    if not os.path.exists("my_annoy_index_and_docstore"):
        annoy_store = Annoy.from_documents(texts, embeddings)
        annoy_store.save_local("my_annoy_index_and_docstore")

    # Check and create FAISS index if not exists
    if not os.path.exists("faiss_index"):
        faiss_store = FAISS.from_documents(texts, embeddings)
        faiss_store.save_local("faiss_index")

    # Check and create Chroma index if not exists
    if not os.path.exists("./chroma_db"):
        chroma_store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
        chroma_store.persist()

def get_conversational_chain(completion_model, temperature):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model=completion_model, temperature=temperature)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def ans_chroma(user_question, embeddings, top_k, completion_model, temperature):
    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    start_time = time.time()
    docs = new_db.similarity_search(user_question, k=top_k)
    chroma_time = time.time() - start_time

    chain = get_conversational_chain(completion_model, temperature)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"], chroma_time, len(docs)

def ans_faiss(user_question, embeddings, top_k, completion_model, temperature):
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    start_time = time.time()
    docs = new_db.similarity_search(user_question, k=top_k)
    faiss_time = time.time() - start_time

    chain = get_conversational_chain(completion_model, temperature)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"], faiss_time, len(docs)

def ans_annoy(user_question, embeddings, top_k, completion_model, temperature):
    new_db = Annoy.load_local("my_annoy_index_and_docstore", embeddings=embeddings, allow_dangerous_deserialization=True)
    start_time = time.time()
    docs = new_db.similarity_search(user_question, k=top_k)
    annoy_time = time.time() - start_time

    chain = get_conversational_chain(completion_model, temperature)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"], annoy_time, len(docs)

def generate_comments(performance_metrics):
    comments = ""
    
    for db, metrics in performance_metrics.items():
        comments += f"For {db}:\n"
        comments += f"- Query Time: {metrics['query_time']:.2f} seconds\n"
        comments += f"- Documents Retrieved: {metrics['num_documents']}\n"
        
        if metrics['query_time'] > 5.0:
            comments += "  - Observation: The query time is relatively high. Consider optimizing the index or the hardware.\n"
        else:
            comments += "  - Observation: The query time is within acceptable limits.\n"

        if metrics['num_documents'] < 3:
            comments += "  - Observation: Fewer documents were retrieved. Ensure the index is correctly built and contains relevant documents.\n"
        else:
            comments += "  - Observation: A sufficient number of documents were retrieved.\n"
    
    return comments

def user_input(user_question, top_k, embedding_model, completion_model, temperature):
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

    # Store question in session state
    st.session_state.questions.append(user_question)

    ans_c, chroma_time, chroma_docs_count = ans_chroma(user_question, embeddings, top_k, completion_model, temperature)
    ans_f, faiss_time, faiss_docs_count = ans_faiss(user_question, embeddings, top_k, completion_model, temperature)
    ans_a, annoy_time, annoy_docs_count = ans_annoy(user_question, embeddings, top_k, completion_model, temperature)

    st.session_state.answers = {
        "chroma": ans_c,
        "faiss": ans_f,
        "annoy": ans_a,
        "chroma_time": chroma_time,
        "faiss_time": faiss_time,
        "annoy_time": annoy_time,
        "chroma_docs_count": chroma_docs_count,
        "faiss_docs_count": faiss_docs_count,
        "annoy_docs_count": annoy_docs_count
    }

    # Performance comparison metrics
    performance_metrics = {
        "Chroma": {
            "query_time": chroma_time,
            "num_documents": chroma_docs_count
        },
        "FAISS": {
            "query_time": faiss_time,
            "num_documents": faiss_docs_count
        },
        "Annoy": {
            "query_time": annoy_time,
            "num_documents": annoy_docs_count
        }
    }

    # Generate comments and observations
    comments = generate_comments(performance_metrics)

    # Store this run's data
    config_settings = {
        "embedding_model": embedding_model,
        "completion_model": completion_model,
        "temperature": temperature,
        "top_k": top_k
    }
    store_data(config_settings, {"Chroma": ans_c, "FAISS": ans_f, "Annoy": ans_a}, performance_metrics, comments)

def display_results():
    if 'answers' in st.session_state:
        st.write("**Replies:**")
        st.write(f"**Chroma:** {st.session_state.answers['chroma']}")
        st.write(f"**FAISS:** {st.session_state.answers['faiss']}")
        st.write(f"**Annoy:** {st.session_state.answers['annoy']}")
        st.write(f"Chroma query time: {st.session_state.answers['chroma_time']:.2f} seconds")
        st.write(f"FAISS query time: {st.session_state.answers['faiss_time']:.2f} seconds")
        st.write(f"Annoy query time: {st.session_state.answers['annoy_time']:.2f} seconds")
        st.write(f"Chroma documents retrieved: {st.session_state.answers['chroma_docs_count']}")
        st.write(f"FAISS documents retrieved: {st.session_state.answers['faiss_docs_count']}")
        st.write(f"Annoy documents retrieved: {st.session_state.answers['annoy_docs_count']}")

        # Display stored data
        st.write("**Stored Data:**")
        for idx, data_entry in enumerate(st.session_state.stored_data):
            st.write(f"**Run {idx + 1}:**")
            st.write(f"Config Settings: {data_entry['config_settings']}")
            st.write(f"Retrieved Nodes: {data_entry['retrieved_nodes']}")
            st.write(f"Performance Metrics: {data_entry['performance_metrics']}")
            st.write(f"Comments and Observations: {data_entry['comments']}")
            st.write("---")

def main():
    st.set_page_config(page_title="Vector Database Comparison System")
    st.header("Vector Database Comparison System 📊")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        
        chunk_size = st.slider("Chunk Size", min_value=1000, max_value=10000, value=5000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=5000, value=3000, step=100)
        
        embedding_model = st.text_input("Embedding Model", value="models/embedding-001")
        completion_model = st.text_input("Completion Model", value="gemini-pro")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5, step=1)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                vector(pdf_docs, chunk_size, chunk_overlap)
                st.success("Done")
    
    user_question = st.text_input("Ask a Question from the PDF Files", key="initial_question")

    if user_question:
        user_input(user_question, top_k, embedding_model, completion_model, temperature)
        display_results()

if __name__ == "__main__":
    main()
