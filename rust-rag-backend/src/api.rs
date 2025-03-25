use actix_web::{
    delete, get, post, web, HttpResponse, Responder, Error as ActixError,
};
use actix_multipart::Multipart;
use anyhow::{anyhow, Result};
use chrono::Utc;
use futures::{StreamExt, TryStreamExt};
use log::{error, info};
use serde_json;
use std::io::Write;
use std::path::Path;
use std::fs;

use crate::db;
use crate::models::{
    AppState, ChatRequest, ChatResponse, ChatHistoryResponse, DocumentAddRequest,
    DocumentAddResponse, DocumentList, ErrorResponse, FileUploadResponse,
    SearchResult, DocumentMetadata
};
use crate::utils;

// Endpoint chatbot dengan RAG
#[post("/chat")]
pub async fn chat(
    app_state: web::Data<AppState>,
    req: web::Json<ChatRequest>,
) -> impl Responder {
    // Parse request
    let query = &req.query;
    
    // Cari dokumen yang relevan
    match retrieve_documents(&app_state, query).await {
        Ok((relevant_docs, metadata)) => {
            // Buat konteks dari dokumen yang relevan
            let mut context = String::new();
            for (i, doc) in relevant_docs.iter().enumerate() {
                let source_info = format!(
                    "[Dokumen: {}]",
                    metadata.get(i)
                        .map(|m| m.title.clone())
                        .unwrap_or_else(|| "Tanpa judul".to_string())
                );
                context.push_str(&format!("{}\n{}\n\n", source_info, doc));
            }
            
            // Buat prompt untuk API DeepSeek
            let prompt = format!(
                "Gunakan informasi berikut untuk menjawab pertanyaan pengguna.\n\
                Jika informasi tidak mencukupi, katakan Anda tidak memiliki cukup informasi.\n\
                Jangan mengarang jawaban jika tidak ada di konteks.\n\
                Sertakan sumber dokumen yang digunakan dalam jawaban.\n\n\
                Konteks:\n{}\n\n\
                Pertanyaan: {}\n\
                Jawaban:",
                context, query
            );
            
            // Panggil DeepSeek API
            match call_llm_api(&prompt).await {
                Ok(response) => {
                    // Simpan ke database
                    if let Err(e) = db::add_chat_history(&app_state.db, query, &response) {
                        error!("❌ Gagal menyimpan riwayat chat: {:?}", e);
                    }
                    
                    HttpResponse::Ok().json(ChatResponse {
                        response,
                    })
                },
                Err(e) => {
                    error!("❌ Gagal memanggil API LLM: {:?}", e);
                    HttpResponse::InternalServerError().json(ErrorResponse {
                        error: "Gagal memanggil API LLM".to_string(),
                    })
                }
            }
        },
        Err(e) => {
            error!("❌ Gagal mengambil dokumen: {:?}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Gagal mengambil dokumen yang relevan".to_string(),
            })
        }
    }
}

// Mencari dokumen relevan
async fn retrieve_documents(
    app_state: &web::Data<AppState>,
    query: &str,
) -> Result<(Vec<String>, Vec<DocumentMetadata>)> {
    // Cek cache
    if let Ok(Some(cache_result)) = db::check_query_cache(&app_state.db, query) {
        // Retrieve dari cache
        let documents: Vec<String> = serde_json::from_str(&cache_result)?;
        
        // Dapatkan metadata
        let mut metadata = Vec::new();
        for doc in &documents {
            if let Ok(Some(doc_metadata)) = db::get_document_metadata(&app_state.db, doc) {
                metadata.push(doc_metadata);
            } else {
                metadata.push(DocumentMetadata {
                    title: "Tanpa judul".to_string(),
                    source: "unknown".to_string(),
                });
            }
        }
        
        return Ok((documents, metadata));
    }
    
    // Jika tidak ada di cache, lakukan pencarian semantik
    let search_result = app_state.embedding.search(query, 3)?;
    
    // Dapatkan metadata
    let mut metadata = Vec::new();
    for doc in &search_result {
        if let Ok(Some(doc_metadata)) = db::get_document_metadata(&app_state.db, doc) {
            metadata.push(doc_metadata);
        } else {
            metadata.push(DocumentMetadata {
                title: "Tanpa judul".to_string(),
                source: "unknown".to_string(),
            });
        }
    }
    
    // Simpan ke cache
    let cache_json = serde_json::to_string(&search_result)?;
    if let Err(e) = db::save_query_cache(&app_state.db, query, &cache_json) {
        error!("❌ Gagal menyimpan cache: {:?}", e);
    }
    
    Ok((search_result, metadata))
}

// Panggil API LLM (DeepSeek)
async fn call_llm_api(prompt: &str) -> Result<String> {
    // Get API key from env
    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .map_err(|_| anyhow!("API key tidak ditemukan"))?;
    
    // Setup client
    let client = reqwest::Client::new();
    
    // Prepare request
    let payload = serde_json::json!({
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    });
    
    // Send request
    let response = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;
    
    // Parse response
    let response_json: serde_json::Value = response.json().await?;
    
    // Extract response text
    let text = response_json
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .ok_or_else(|| anyhow!("Format respons tidak valid"))?
        .to_string();
    
    Ok(text)
}

// Endpoint untuk menambahkan dokumen
#[post("/add_document")]
pub async fn add_document(
    app_state: web::Data<AppState>,
    req: web::Json<DocumentAddRequest>,
) -> impl Responder {
    // Preprocess teks
    let text = utils::preprocess_text(&req.text);
    let title = req.title.clone().unwrap_or_else(|| {
        format!("Dokumen {}", Utc::now().format("%Y-%m-%d %H:%M:%S"))
    });
    
    // Buat embedding
    match app_state.embedding.encode(&text) {
        Ok(embedding) => {
            // Simpan ke database
            match db::add_document(&app_state.db, &text, &embedding, &title, "manual_input") {
                Ok(doc_id) => {
                    // Tambahkan dokumen ke index
                    if let Err(e) = app_state.embedding.add_document(&text, &doc_id) {
                        error!("❌ Gagal menambahkan dokumen ke HNSW index: {:?}", e);
                    }
                    
                    HttpResponse::Ok().json(DocumentAddResponse {
                        message: "Dokumen berhasil ditambahkan!".to_string(),
                        id: doc_id,
                    })
                },
                Err(e) => {
                    error!("❌ Gagal menyimpan dokumen: {:?}", e);
                    HttpResponse::InternalServerError().json(ErrorResponse {
                        error: "Gagal menyimpan dokumen".to_string(),
                    })
                }
            }
        },
        Err(e) => {
            error!("❌ Gagal membuat embedding: {:?}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Gagal membuat embedding".to_string(),
            })
        }
    }
}

// Endpoint untuk mengunggah file
#[post("/upload_file")]
pub async fn upload_file(
    app_state: web::Data<AppState>,
    mut payload: Multipart,
) -> Result<HttpResponse, ActixError> {
    // Check if temp directory exists, create if necessary
    let temp_dir = Path::new("./temp");
    if !temp_dir.exists() {
        fs::create_dir_all(temp_dir).map_err(|e| {
            error!("❌ Gagal membuat direktori temp: {:?}", e);
            actix_web::error::ErrorInternalServerError("Gagal membuat direktori temp")
        })?;
    }
    
    // Process payload
    while let Ok(Some(mut field)) = payload.try_next().await {
        // Get field info
        let content_disp = field.content_disposition();
        let filename = content_disp
            .get_filename()
            .ok_or_else(|| {
                error!("❌ Filename tidak ditemukan");
                actix_web::error::ErrorBadRequest("Filename tidak ditemukan")
            })?
            .to_string();
        
        // Check file extension
        let ext = utils::get_file_extension(&filename.to_lowercase())
            .ok_or_else(|| {
                error!("❌ Ekstensi file tidak valid");
                actix_web::error::ErrorBadRequest("Ekstensi file tidak valid")
            })?;
        
        // Create temp file
        let file_path = temp_dir.join(&filename);
        let mut file = fs::File::create(&file_path).map_err(|e| {
            error!("❌ Gagal membuat file temp: {:?}", e);
            actix_web::error::ErrorInternalServerError("Gagal membuat file temp")
        })?;
        
        // Write to temp file
        while let Some(chunk) = field.next().await {
            let data = chunk.map_err(|e| {
                error!("❌ Gagal membaca chunk file: {:?}", e);
                actix_web::error::ErrorInternalServerError("Gagal membaca chunk file")
            })?;
            file.write_all(&data).map_err(|e| {
                error!("❌ Gagal menulis ke file temp: {:?}", e);
                actix_web::error::ErrorInternalServerError("Gagal menulis ke file temp")
            })?;
        }
        
        // Read file content
        let file_data = fs::read(&file_path).map_err(|e| {
            error!("❌ Gagal membaca file: {:?}", e);
            actix_web::error::ErrorInternalServerError("Gagal membaca file")
        })?;
        
        // Extract text based on file type
        let (text, source_type) = match ext {
            "pdf" => {
                let text = utils::extract_pdf_text(&file_data).map_err(|e| {
                    error!("❌ Gagal ekstrak PDF: {:?}", e);
                    actix_web::error::ErrorInternalServerError("Gagal ekstrak PDF")
                })?;
                (text, "pdf")
            },
            "csv" => {
                let text = utils::extract_csv_text(&file_data).map_err(|e| {
                    error!("❌ Gagal ekstrak CSV: {:?}", e);
                    actix_web::error::ErrorInternalServerError("Gagal ekstrak CSV")
                })?;
                (text, "csv")
            },
            "txt" => {
                let text = utils::extract_txt_text(&file_data).map_err(|e| {
                    error!("❌ Gagal ekstrak TXT: {:?}", e);
                    actix_web::error::ErrorInternalServerError("Gagal ekstrak TXT")
                })?;
                (text, "txt")
            },
            _ => {
                return Err(actix_web::error::ErrorBadRequest("Format file tidak didukung"));
            }
        };
        
        // Process text
        let processed_text = utils::preprocess_text(&text);
        let title = Path::new(&filename).file_name().unwrap().to_string_lossy().to_string();
        
        // Generate embedding
        let embedding = app_state.embedding.encode(&processed_text).map_err(|e| {
            error!("❌ Gagal membuat embedding: {:?}", e);
            actix_web::error::ErrorInternalServerError("Gagal membuat embedding")
        })?;
        
        // Save to database
        let doc_id = db::add_document(
            &app_state.db,
            &processed_text,
            &embedding,
            &title,
            source_type
        ).map_err(|e| {
            error!("❌ Gagal menyimpan dokumen: {:?}", e);
            actix_web::error::ErrorInternalServerError("Gagal menyimpan dokumen")
        })?;
        
        // Add to HNSW index
        if let Err(e) = app_state.embedding.add_document(&processed_text, &doc_id) {
            error!("❌ Gagal menambahkan dokumen ke HNSW index: {:?}", e);
        }
        
        // Clean up temp file
        if let Err(e) = fs::remove_file(&file_path) {
            error!("⚠️ Gagal menghapus file temp: {:?}", e);
        }
        
        // Return success
        return Ok(HttpResponse::Ok().json(FileUploadResponse {
            message: "File berhasil diproses dan disimpan!".to_string(),
            filename,
            id: doc_id,
        }));
    }
    
    // If we get here, no file was processed
    Err(actix_web::error::ErrorBadRequest("Tidak ada file yang diunggah"))
}

// Endpoint untuk mendapatkan dokumen
#[get("/documents")]
pub async fn get_documents(
    app_state: web::Data<AppState>,
) -> impl Responder {
    match db::get_documents(&app_state.db) {
        Ok(documents) => {
            HttpResponse::Ok().json(DocumentList { documents })
        },
        Err(e) => {
            error!("❌ Gagal mengambil dokumen: {:?}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Gagal mengambil dokumen".to_string(),
            })
        }
    }
}

// Endpoint untuk menghapus dokumen
#[delete("/documents/{doc_id}")]
pub async fn delete_document(
    app_state: web::Data<AppState>,
    path: web::Path<String>,
) -> impl Responder {
    let doc_id = path.into_inner();
    
    // Hapus dari database
    if let Err(e) = db::delete_document(&app_state.db, &doc_id) {
        error!("❌ Gagal menghapus dokumen dari database: {:?}", e);
        return HttpResponse::InternalServerError().json(ErrorResponse {
            error: "Gagal menghapus dokumen dari database".to_string(),
        });
    }
    
    // Hapus dari HNSW index
    if let Err(e) = app_state.embedding.delete_document(&doc_id) {
        error!("❌ Gagal menghapus dokumen dari HNSW index: {:?}", e);
    }
    
    HttpResponse::Ok().json(serde_json::json!({
        "message": format!("Dokumen dengan ID {} berhasil dihapus", doc_id)
    }))
}

// Endpoint untuk mendapatkan riwayat chat
#[get("/chat_history")]
pub async fn get_chat_history(
    app_state: web::Data<AppState>,
    query: web::Query<std::collections::HashMap<String, String>>,
) -> impl Responder {
    // Parse limit parameter
    let limit = query.get("limit")
        .and_then(|l| l.parse::<usize>().ok())
        .unwrap_or(10);
    
    match db::get_chat_history(&app_state.db, limit) {
        Ok(history) => {
            HttpResponse::Ok().json(ChatHistoryResponse { history })
        },
        Err(e) => {
            error!("❌ Gagal mengambil riwayat chat: {:?}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Gagal mengambil riwayat chat".to_string(),
            })
        }
    }
} 