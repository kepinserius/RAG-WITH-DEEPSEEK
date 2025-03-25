use actix_cors::Cors;
use actix_web::{web, App, HttpServer, middleware, Result};
use dotenv::dotenv;
use log::{info, warn};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;

mod api;
mod db;
mod embedding;
mod models;
mod utils;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("🚀 Menginisialisasi RAG Chatbot Backend");
    
    // Inisialisasi database
    let db_pool = match db::init_db_pool() {
        Ok(pool) => {
            info!("✅ Koneksi database berhasil dibuat");
            pool
        },
        Err(e) => {
            panic!("❌ Tidak dapat menginisialisasi database: {:?}", e);
        }
    };
    
    // Setup database tables
    if let Err(e) = db::setup_database(&db_pool) {
        panic!("❌ Gagal setup database: {:?}", e);
    }
    
    // Inisialisasi model embedding
    let embedding_model = match embedding::init_model() {
        Ok(model) => {
            info!("✅ Model embedding berhasil dimuat");
            model
        },
        Err(e) => {
            panic!("❌ Tidak dapat menginisialisasi model embedding: {:?}", e);
        }
    };
    
    // Siapkan shared state
    let app_state = web::Data::new(models::AppState {
        db: db_pool,
        embedding: Arc::new(embedding_model),
    });
    
    // Setup server
    let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = env::var("PORT").unwrap_or_else(|_| "5000".to_string());
    let server_url = format!("{}:{}", host, port);
    
    info!("🌐 Server berjalan di http://{}", server_url);
    
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);
        
        App::new()
            .wrap(middleware::Logger::default())
            .wrap(cors)
            .app_data(app_state.clone())
            // API routes
            .service(api::chat)
            .service(api::add_document)
            .service(api::upload_file)
            .service(api::get_documents)
            .service(api::delete_document)
            .service(api::get_chat_history)
    })
    .bind(server_url)?
    .run()
    .await
} 