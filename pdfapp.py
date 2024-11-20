import streamlit as slt
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to build vector store
def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore.save_local("faiss_index")

# Function to build conversation chain
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function for user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    slt.write("Reply: ", response["output_text"])

# Main Streamlit app
def main():
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    slt.set_page_config(
        page_title="Your AI Chat Assistant for Seamless Conversations",
        page_icon=":book:",
    )
    slt.header("Chat with PDF :book:")
    user_question=slt.text_input("Ask questions about your documents")
    if user_question:
        user_input(user_question)

    with slt.sidebar:
        slt.header("Your Documents")
        pdf_docs = slt.file_uploader(
            "Upload your documents and click 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if pdf_docs and slt.button("Process"):
            with slt.spinner("Processing..."):
                try:
                    # Step 1: Load the documents
                    slt.write("Loading documents...")
                    raw_text = get_pdf_text(pdf_docs)

                    # Step 2: Create chunks from the text
                    slt.write("Creating text chunks...")
                    chunks_pdf = get_text_chunks(raw_text)

                    # Step 3: Build vector store
                    slt.write("Building vector store...")
                    get_vectorstore(chunks_pdf)

                    slt.success("Process completed successfully! ✅")
                except Exception as e:
                    slt.error(f"An error occurred: {str(e)}")
        elif not pdf_docs:
            slt.info("Please upload at least one PDF document to process.")


if __name__ == "__main__":
    main()
