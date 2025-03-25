use crate::embedding::EmbeddingModel;
use chrono::{DateTime, Utc};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Definisi tipe database pool
pub type DbPool = Pool<SqliteConnectionManager>;

// Shared state aplikasi
pub struct AppState {
    pub db: DbPool,
    pub embedding: Arc<EmbeddingModel>,
}

// Struktur untuk chat
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub query: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub response: String,
}

// Struktur untuk riwayat chat
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatHistory {
    pub query: String,
    pub response: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatHistoryResponse {
    pub history: Vec<ChatHistory>,
}

// Struktur untuk dokumen
#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub text: String,
    pub source: String,
    pub preview: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentList {
    pub documents: Vec<Document>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentAddRequest {
    pub text: String,
    pub title: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentAddResponse {
    pub message: String,
    pub id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileUploadResponse {
    pub message: String,
    pub filename: String,
    pub id: String,
}

// Struktur untuk metadata dokumen
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DocumentMetadata {
    pub title: String,
    pub source: String,
}

// Struktur untuk hasil pencarian
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub documents: Vec<String>,
    pub metadata: Vec<DocumentMetadata>,
}

// Struktur pesan error
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
} 