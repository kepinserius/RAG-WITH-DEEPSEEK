import os
import faiss
import chromadb
import openai  # Menggunakan OpenAI library untuk DeepSeek
import fitz  # PyMuPDF untuk PDF
import pandas as pd
import redis
import psycopg2
import pymongo
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

# Konfigurasi API DeepSeek
os.environ["DEEPSEEK_API_KEY"] = "your_api_key_here"
openai.api_key = os.getenv("DEEPSEEK_API_KEY")

# Load model embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Inisialisasi ChromaDB
chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(name="documents")

# Inisialisasi Redis untuk caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Inisialisasi PostgreSQL untuk history chat
conn = psycopg2.connect(database="rag_chat", user="user", password="password", host="localhost", port="5432")
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        query TEXT,
        response TEXT
    )
""")
conn.commit()

# Inisialisasi MongoDB untuk menyimpan dokumen
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["rag_db"]
doc_collection = db["documents"]

# Inisialisasi Flask
app = Flask(__name__)

# Fungsi preprocessing teks
def preprocess_text(text):
    return text.strip().replace("\n", " ")

# Endpoint untuk menambahkan dokumen ke database
@app.route("/add_document", methods=["POST"])
def add_document():
    data = request.json
    text = preprocess_text(data["text"])

    # Embedding teks
    vector = embedding_model.encode([text])[0].tolist()

    # Simpan ke ChromaDB
    collection.add(documents=[text], embeddings=[vector], ids=[str(len(collection.get()["ids"]))])
    
    # Simpan ke MongoDB
    doc_collection.insert_one({"text": text, "embedding": vector})

    return jsonify({"message": "Dokumen berhasil ditambahkan!"})

# Endpoint untuk mengunggah file PDF atau CSV
@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files["file"]
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
    elif filename.endswith(".csv"):
        df = pd.read_csv(file)
        text = "\n".join(df.astype(str).values.flatten())
    else:
        return jsonify({"error": "Format file tidak didukung!"})

    text = preprocess_text(text)

    # Embedding dan simpan ke database
    vector = embedding_model.encode([text])[0].tolist()
    collection.add(documents=[text], embeddings=[vector], ids=[str(len(collection.get()["ids"]))])
    doc_collection.insert_one({"text": text, "embedding": vector})

    return jsonify({"message": "File berhasil diproses dan disimpan!"})

# Fungsi untuk mencari dokumen relevan (dengan caching)
def retrieve_documents(query):
    cached_result = redis_client.get(query)
    if cached_result:
        return cached_result.decode("utf-8").split("\n")
    
    query_vector = embedding_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    
    documents = results["documents"][0] if results["documents"] else []
    redis_client.set(query, "\n".join(documents), ex=3600)  # Cache selama 1 jam
    
    return documents

# Endpoint chatbot dengan RAG + penyimpanan ke PostgreSQL
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data["query"]

    # Cari dokumen yang relevan
    relevant_docs = retrieve_documents(user_query)
    context = "\n".join(relevant_docs)

    # Prompt untuk model DeepSeek
    prompt = f"Gunakan informasi berikut untuk menjawab pertanyaan pengguna:\n{context}\n\nPertanyaan: {user_query}\nJawaban:"

    # Panggil DeepSeek API menggunakan OpenAI library
    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )["choices"][0]["message"]["content"]

    # Simpan ke PostgreSQL
    cur.execute("INSERT INTO chat_history (query, response) VALUES (%s, %s)", (user_query, response))
    conn.commit()

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
