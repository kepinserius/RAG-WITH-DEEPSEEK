use crate::models::{ChatHistory, DbPool, Document, DocumentMetadata};
use anyhow::{Context, Result};
use log::{error, info};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use serde_json;
use std::path::Path;
use std::fs;

// Inisialisasi pool koneksi database
pub fn init_db_pool() -> Result<DbPool> {
    let db_path = std::env::var("DB_PATH").unwrap_or_else(|_| "ragchat.db".to_string());
    
    // Buat direktori induk jika belum ada
    if let Some(parent) = Path::new(&db_path).parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    
    let manager = SqliteConnectionManager::file(&db_path);
    let pool = Pool::new(manager)
        .context("Gagal membuat pool koneksi SQLite")?;
    
    Ok(pool)
}

// Setup tabel-tabel database
pub fn setup_database(pool: &DbPool) -> Result<()> {
    let conn = pool.get()?;
    
    // Tabel riwayat chat
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    
    // Tabel cache query
    conn.execute(
        "CREATE TABLE IF NOT EXISTS query_cache (
            query TEXT PRIMARY KEY,
            results TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    
    // Tabel dokumen
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    
    info!("✅ Database setup selesai");
    Ok(())
}

// Tambahkan dokumen ke database
pub fn add_document(pool: &DbPool, text: &str, embedding: &[f32], title: &str, source: &str) -> Result<String> {
    let conn = pool.get()?;
    let embedding_json = serde_json::to_string(embedding)?;
    
    conn.execute(
        "INSERT INTO documents (text, embedding, title, source) VALUES (?, ?, ?, ?)",
        params![text, embedding_json, title, source],
    )?;
    
    let id = conn.last_insert_rowid().to_string();
    
    Ok(id)
}

// Dapatkan semua dokumen
pub fn get_documents(pool: &DbPool) -> Result<Vec<Document>> {
    let conn = pool.get()?;
    let mut stmt = conn.prepare("SELECT id, text, title, source FROM documents")?;
    
    let docs = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let text: String = row.get(1)?;
        let title: String = row.get(2)?;
        let source: String = row.get(3)?;
        
        let preview = if text.len() > 100 {
            format!("{}...", &text[..100])
        } else {
            text.clone()
        };
        
        Ok(Document {
            id: id.to_string(),
            text,
            title,
            source,
            preview: Some(preview),
        })
    })?;
    
    let mut documents = Vec::new();
    for doc in docs {
        documents.push(doc?);
    }
    
    Ok(documents)
}

// Hapus dokumen berdasarkan ID
pub fn delete_document(pool: &DbPool, doc_id: &str) -> Result<()> {
    let conn = pool.get()?;
    let doc_id = doc_id.parse::<i64>()?;
    
    conn.execute("DELETE FROM documents WHERE id = ?", params![doc_id])?;
    
    Ok(())
}

// Tambahkan riwayat chat
pub fn add_chat_history(pool: &DbPool, query: &str, response: &str) -> Result<()> {
    let conn = pool.get()?;
    
    conn.execute(
        "INSERT INTO chat_history (query, response) VALUES (?, ?)",
        params![query, response],
    )?;
    
    Ok(())
}

// Dapatkan riwayat chat
pub fn get_chat_history(pool: &DbPool, limit: usize) -> Result<Vec<ChatHistory>> {
    let conn = pool.get()?;
    let mut stmt = conn.prepare(
        "SELECT query, response FROM chat_history 
         ORDER BY timestamp DESC LIMIT ?",
    )?;
    
    let history = stmt.query_map(params![limit], |row| {
        let query: String = row.get(0)?;
        let response: String = row.get(1)?;
        
        Ok(ChatHistory { query, response })
    })?;
    
    let mut chat_history = Vec::new();
    for entry in history {
        chat_history.push(entry?);
    }
    
    Ok(chat_history)
}

// Cek cache untuk query
pub fn check_query_cache(pool: &DbPool, query: &str) -> Result<Option<String>> {
    let conn = pool.get()?;
    
    let mut stmt = conn.prepare("SELECT results FROM query_cache WHERE query = ?")?;
    let mut rows = stmt.query(params![query])?;
    
    if let Some(row) = rows.next()? {
        let results: String = row.get(0)?;
        Ok(Some(results))
    } else {
        Ok(None)
    }
}

// Simpan hasil query ke cache
pub fn save_query_cache(pool: &DbPool, query: &str, results: &str) -> Result<()> {
    let conn = pool.get()?;
    
    conn.execute(
        "INSERT OR REPLACE INTO query_cache (query, results) VALUES (?, ?)",
        params![query, results],
    )?;
    
    Ok(())
}

// Dapatkan metadata dokumen berdasarkan teks
pub fn get_document_metadata(pool: &DbPool, text: &str) -> Result<Option<DocumentMetadata>> {
    let conn = pool.get()?;
    
    let mut stmt = conn.prepare("SELECT title, source FROM documents WHERE text = ? LIMIT 1")?;
    let mut rows = stmt.query(params![text])?;
    
    if let Some(row) = rows.next()? {
        let title: String = row.get(0)?;
        let source: String = row.get(1)?;
        
        Ok(Some(DocumentMetadata { title, source }))
    } else {
        Ok(None)
    }
} 