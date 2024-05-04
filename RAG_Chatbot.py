from langchain_community.llms import Ollama
from flask import Flask, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)
llm = Ollama(model="llama3")
VECTOR_STORE_PATH = "vector_db"

embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

base_template = PromptTemplate.from_template(
    """
    
    You are a technical assistant good at searching documents. Your task is to answer based on the context provided to
    you. If you don't know the answer say "Nahi pata bhaiya sorry"
    [INST] {input}
            Context: {context}
            Answer: 
    [/INST]
    """
)

@app.route("/ai", methods=["POST"])
def ai_post():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = llm.invoke(query)

    return response


@app.route("/ask_pdf", methods=["POST"])
def ai_pdf_post():
    print("Post /ask pdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    print('loading verctor store')
    vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type = 'similarity_score_threshold',
        search_kwargs = {
            'k': 20,
            'score_threshold': 0.1
        }
    )

    chain = load_qa_chain(llm=llm, chain_type="stuff")

    response = llm.invoke(query)

    return response


@app.route("/pdf_upload", methods=["POST"])
def pdf_post():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf_files/" + file_name
    file.save(save_file)
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len = {len(docs)}")
    chunks = text_splitter.split_documents(docs)
    print(f"docs len = {len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=VECTOR_STORE_PATH
    )
    vector_store.persist()

    print(f"filename: {file_name}")
    response = {
        "status": "Successfully uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
