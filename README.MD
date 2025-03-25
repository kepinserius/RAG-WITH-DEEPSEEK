# RAG Chatbot

Aplikasi chatbot dengan menggunakan Retrieval Augmented Generation (RAG).

## Fitur

- 🤖 Chatbot berbasis RAG menggunakan model DeepSeek
- 📄 Dukungan unggah dokumen (PDF, CSV, TXT)
- 📝 Manajemen dokumen (tambah, hapus, lihat)
- 💾 Penyimpanan chat history
- 🔍 Pencarian semantik untuk menemukan dokumen yang relevan

## Struktur Proyek

```
/
├── backend.py           # API Flask untuk backend
├── ragchat.db           # Database SQLite untuk penyimpanan data
├── chatbot-rag/         # Frontend Next.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/
│   │   │   │   ├── ChatBox.tsx         # Komponen chat
│   │   │   │   ├── DocumentManager.tsx # Komponen manajemen dokumen
│   │   │   │   └── Navigation.tsx      # Komponen navigasi
│   │   │   ├── documents/
│   │   │   │   └── page.tsx            # Halaman dokumen
│   │   │   ├── page.tsx                # Halaman utama (chat)
│   │   │   └── layout.tsx              # Layout utama
├── db/                  # Database ChromaDB lokal
```

## Teknologi yang Digunakan

### Backend

- Flask: Web Framework
- ChromaDB: Database vektor untuk penyimpanan dan pencarian semantik
- SentenceTransformers: Model embedding untuk pencarian semantik
- PyMuPDF: Ekstraksi teks dari PDF
- SQLite: Database untuk menyimpan riwayat chat, dokumen, dan cache

### Frontend

- Next.js: React framework
- Tailwind CSS: Framework CSS untuk styling

## Cara Menjalankan

### Backend

1. Buat virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. Install dependencies

```bash
pip install flask faiss-cpu chromadb openai pymupdf pandas sqlite3 sentence-transformers flask-cors
```

3. Jalankan backend

```bash
python backend.py
```

### Frontend

1. Masuk ke direktori frontend

```bash
cd chatbot-rag
```

2. Install dependencies

```bash
npm install
```

3. Jalankan frontend

```bash
npm run dev
```

4. Buka browser dan akses http://localhost:3000

## Endpoint API

- `POST /chat`: Mengirim pertanyaan ke chatbot
- `POST /add_document`: Menambahkan dokumen secara manual
- `POST /upload_file`: Mengunggah dokumen (PDF/CSV/TXT)
- `GET /documents`: Mendapatkan daftar dokumen
- `DELETE /documents/<id>`: Menghapus dokumen
- `GET /chat_history`: Mendapatkan riwayat chat

## Penyimpanan Data

Aplikasi ini menggunakan dua metode penyimpanan:

1. **ChromaDB**: Database vektor berbasis file yang digunakan untuk penyimpanan teks dokumen dan melakukan pencarian semantik/vektor
2. **SQLite**: Database SQL ringan berbasis file (ragchat.db) yang digunakan untuk:
   - Menyimpan riwayat chat
   - Caching hasil pencarian
   - Menyimpan metadata dan informasi dokumen

Keuntungan menggunakan SQLite:

- Tidak memerlukan server database terpisah
- Data disimpan dalam file tunggal yang mudah dibackup
- Mudah digunakan untuk pengembangan dan aplikasi dengan beban rendah

## Cara Kerja RAG

1. Dokumen diunggah dan disimpan dalam database vektor (ChromaDB) dan SQLite.
2. Saat pengguna mengirim pertanyaan, sistem mencari dokumen yang relevan.
3. Dokumen yang relevan digunakan sebagai konteks untuk model LLM.
4. Model LLM (DeepSeek) menghasilkan jawaban berdasarkan konteks + pertanyaan.
