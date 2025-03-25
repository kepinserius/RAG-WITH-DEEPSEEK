# RAG Chatbot (Backend Rust)

Backend berbasis Rust untuk aplikasi chatbot dengan Retrieval Augmented Generation (RAG).

## Keuntungan Penggunaan Rust

1. **Performa Tinggi**: Eksekusi mendekati kecepatan C/C++ tanpa garbage collector
2. **Keamanan Memori**: Sistem ownership dan borrow checker mencegah memory leaks dan race conditions
3. **Konkurensi Aman**: Fitur konkurensi yang aman berdasarkan tipe, tanpa data races
4. **Zero-Cost Abstractions**: Abstraksi tingkat tinggi tanpa overhead runtime
5. **Peningkatan Skalabilitas**: Penanganan beban tinggi dengan penggunaan resource yang efisien

## Fitur

- 🤖 API chatbot berbasis RAG
- 📄 Dukungan unggah dokumen (PDF, CSV, TXT)
- 📝 Manajemen dokumen (tambah, hapus, lihat)
- 💾 Penyimpanan chat history
- 🔍 Pencarian semantik dengan HNSW untuk menemukan dokumen relevan
- ⚡ Caching untuk meningkatkan performa

## Prasyarat

- [Rust](https://www.rust-lang.org/tools/install) (versi 1.57.0 atau lebih baru)
- [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) (biasanya terinstal dengan Rust)
- [API Key DeepSeek](https://deepseek.com/)

## Instalasi

1. Clone repositori ini:

```bash
git clone <repository_url>
cd rust-rag-backend
```

2. Salin file `.env.example` menjadi `.env` dan sesuaikan:

```bash
cp .env.example .env
```

3. Edit file `.env` dan masukkan API key DeepSeek Anda:

```
DEEPSEEK_API_KEY=your_api_key_here
```

4. Build proyek dengan Cargo:

```bash
cargo build --release
```

## Menjalankan Aplikasi

```bash
cargo run --release
```

Server akan berjalan di `http://0.0.0.0:5000`.

## Endpoint API

- `POST /chat`: Mengirim pertanyaan ke chatbot
- `POST /add_document`: Menambahkan dokumen secara manual
- `POST /upload_file`: Mengunggah dokumen (PDF/CSV/TXT)
- `GET /documents`: Mendapatkan daftar dokumen
- `DELETE /documents/{id}`: Menghapus dokumen
- `GET /chat_history`: Mendapatkan riwayat chat

## Struktur Penyimpanan Data

Aplikasi ini menggunakan dua metode penyimpanan:

1. **HNSW (Hierarchical Navigable Small World)**: Library pencarian vektor dengan kompleksitas logaritmik, digunakan untuk pencarian semantik cepat
2. **SQLite**: Database SQL ringan untuk menyimpan:
   - Riwayat chat
   - Caching hasil pencarian
   - Metadata dan informasi dokumen

## Cara Kerja RAG

1. Dokumen diunggah dan disimpan dalam database SQLite dan index HNSW
2. Saat pengguna mengirim pertanyaan, sistem mencari dokumen yang relevan menggunakan HNSW
3. Dokumen yang relevan digunakan sebagai konteks untuk model LLM
4. Model DeepSeek menghasilkan jawaban berdasarkan konteks + pertanyaan

## Pengembangan

### Struktur Proyek

```
/
├── src/
│   ├── main.rs          # Entry point aplikasi
│   ├── api.rs           # Handler endpoint API
│   ├── db.rs            # Operasi database
│   ├── embedding.rs     # Model embedding dan pencarian semantik
│   ├── models.rs        # Definisi struktur data
│   └── utils.rs         # Fungsi-fungsi utilitas
├── .env                 # File konfigurasi
├── Cargo.toml           # Definisi dependencies
└── README.md            # Dokumentasi
```

### Menjalankan dalam Mode Development

```bash
cargo run
```

## Integrasi dengan Frontend

Backend ini kompatibel dengan aplikasi frontend RAG Chatbot yang dibuat dengan Next.js. Pastikan frontend dikonfigurasi untuk menghubungi backend Rust di `http://localhost:5000`.
