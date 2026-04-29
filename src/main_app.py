import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API Key not found. Please set it in the .env file.")
    raise ValueError("Groq API Key not found.")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# -----------------------------
# LOAD KNOWLEDGE BASE FILES
# -----------------------------
def load_knowledge_base():
    documents = []

    files = [
        "data/zomato_context.txt",
        "data/rca_framework.txt",
        "data/interview_cases.txt"
    ]

    for file in files:
        if not os.path.exists(file):
            st.error(f"Missing required file: {file}")
            raise FileNotFoundError(f"File not found: {file}")

        loader = TextLoader(file, autodetect_encoding=True)
        documents.extend(loader.load())

    return documents


# -----------------------------
# INTERVIEWER PROMPT
# -----------------------------
interviewer_prompt = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template="""
You are an experienced Product Manager at Zomato conducting a Root Cause Analysis interview.

Your job:
1. Acknowledge the candidate's ideas briefly
2. Ask 2–3 specific follow-up questions
3. Guide them through structured RCA
4. Challenge assumptions constructively
5. Push deeper into prioritization and validation

Previous conversation:
{chat_history}

Candidate's latest response:
{question}

Relevant context:
{context}

Respond like a real interviewer.
"""
)


# -----------------------------
# INITIALIZE VECTOR STORE
# -----------------------------
def initialize_system():
    documents = load_knowledge_base()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("chroma_db") and os.listdir("chroma_db"):
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

    return vectorstore


# -----------------------------
# CASE DESCRIPTIONS
# -----------------------------
case_descriptions = {
    "Lunch Conversion Rate Drop":
        "Zomato has observed that lunch order conversion rate is only 2% compared to 10% for breakfast and dinner. Why might this be happening and how would you address it?",

    "Delivery Complaints":
        "There has been a sudden 30% increase in complaints about partial order deliveries. What could be causing this and how would you solve it?",

    "Restaurant Rating Decline":
        "Average restaurant ratings in Pune have dropped 10% over the last month. How would you investigate and address this issue?",

    "User Retention Problem":
        "Weekly active users in Bangalore have declined by 15% despite increased marketing spend. What could be the root causes?",

    "Delivery Time Increases":
        "Average delivery time has increased from 28 to 35 minutes in Delhi NCR region. How would you approach this problem?"
}


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(
        page_title="Zomato PM Interview Simulator",
        page_icon="🍱",
        layout="wide"
    )

    st.markdown(
        "<h1 style='text-align: center; color: #FF5733;'>🍱 Zomato PM Interview Simulator</h1>",
        unsafe_allow_html=True
    )

    st.markdown("""
        <style>
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    try:
        vectorstore = initialize_system()
    except Exception as e:
        st.error(f"Failed to initialize the system: {str(e)}")
        st.error("Please check the knowledge base files.")
        return

    # -----------------------------
    # GROQ LLM
    # -----------------------------
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY
    )

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": interviewer_prompt},
        get_chat_history=lambda h: h,
        verbose=True,
    )

    # -----------------------------
    # SIDEBAR
    # -----------------------------
    with st.sidebar:
        st.header("Interview Settings")

        cases = list(case_descriptions.keys())
        selected_case = st.selectbox(
            "Choose a case:",
            cases,
            index=0
        )

        if st.button("Restart Interview"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.rerun()

    # -----------------------------
    # DEFAULT OPENING MESSAGE
    # -----------------------------
    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"Welcome to your Zomato Product Manager interview! "
                f"I'll be evaluating your approach to Root Cause Analysis (RCA) today.\n\n"

                f"Case: {case_descriptions[selected_case]}\n\n"

                f"Please begin by:\n"
                f"1. Defining the problem as you understand it\n"
                f"2. What initial data would you want to look at?\n"
                f"3. What are your first hypotheses about potential causes?\n\n"

                f"Take your time to structure your thoughts before responding."
            )
        }]

    # -----------------------------
    # CHAT HISTORY
    # -----------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -----------------------------
    # USER INPUT
    # -----------------------------
    if prompt := st.chat_input("Your response"):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            response = chain.invoke({
                "question": prompt
            })

            assistant_message = response["answer"]

        except Exception as e:
            assistant_message = f"System error: {str(e)}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        st.rerun()


if __name__ == "__main__":
    main()