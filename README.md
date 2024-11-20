# CHAT-WITH-PDF-USING-GEMINI

This project is a Streamlit-based web application that enables users to interact with the content of PDF documents using natural language queries. By leveraging Google Generative AI and LangChain, the application processes uploaded PDFs, extracts their content, and creates a vector-based search mechanism for efficient information retrieval.

## Text Extraction:
* This process involves extracting raw text from the uploaded PDF documents.
* In this project, the PyPDF2 library is used for reading and extracting text from each page of the PDFs.

## VectorStore:
A VectorStore is a database designed to store text embeddings for similarity search.
### In this project:
* The extracted text is split into manageable chunks using the CharacterTextSplitter class.
* The GoogleGenerativeAIEmbeddings model generates embeddings (numerical representations) for these text chunks.
* The FAISS (Facebook AI Similarity Search) library stores these embeddings, enabling fast similarity searches.

## Chain Load (Conversational Chain):
The chain handles the logic for answering user questions based on the context retrieved from the VectorStore.
### In this project:
* A custom prompt template ensures the AI provides accurate and context-specific responses.
* The ChatGoogleGenerativeAI model powers the response generation.
* The load_qa_chain function orchestrates the interaction between the retrieved documents and the generative model.
  
## User Experience:
* Add more detailed responses in the Streamlit UI, such as displaying matched document sections.
![1](https://github.com/user-attachments/assets/2bbddba8-e139-4f5c-ae52-df4dcb61f44d)
![2](https://github.com/user-attachments/assets/4bc5a0de-8a0b-4882-8e95-a4022cd6380f)

