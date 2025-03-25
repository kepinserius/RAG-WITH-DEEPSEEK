use anyhow::{Context, Result};
use hnswlib_rs::{Hnsw, SearchType};
use log::{info, warn};
use ndarray::{Array1, Array2};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde_json;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::fs;
use std::collections::HashMap;

pub struct EmbeddingModel {
    model: Arc<Mutex<rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel>>,
    index: Arc<Mutex<Option<Hnsw<f32>>>>,
    document_map: Arc<Mutex<HashMap<usize, String>>>,
    next_id: Arc<Mutex<usize>>,
}

impl EmbeddingModel {
    pub fn new() -> Result<Self> {
        // Inisialisasi model embedding sentence-transformers
        let model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL6V2
        )
        .create_model()
        .context("Gagal memuat model embedding")?;
        
        info!("✅ Model embedding berhasil dimuat");
        
        // Pastikan direktori data ada
        let data_dir = Path::new("./db");
        if !data_dir.exists() {
            fs::create_dir_all(data_dir)?;
        }
        
        // Inisialisasi HNSW index
        let index_path = data_dir.join("hnsw_index.json");
        let mut index = None;
        let dimension = 384; // dimensi untuk all-MiniLM-L6-v2
        
        if index_path.exists() {
            match fs::read_to_string(&index_path) {
                Ok(content) => {
                    match serde_json::from_str::<Hnsw<f32>>(&content) {
                        Ok(loaded_index) => {
                            info!("✅ Index HNSW berhasil dimuat dari {:?}", index_path);
                            index = Some(loaded_index);
                        },
                        Err(e) => {
                            warn!("⚠️ Gagal memuat index HNSW dari {:?}: {:?}", index_path, e);
                            warn!("⚠️ Akan membuat index baru");
                        }
                    }
                },
                Err(e) => {
                    warn!("⚠️ Gagal membaca file index: {:?}", e);
                }
            }
        }
        
        if index.is_none() {
            // Buat index baru jika tidak ada
            let new_index = Hnsw::new(
                dimension,
                1000, // ukuran maksimum
                16,   // connections per layer
                100,  // ef_construction
                200,  // ef (search accuracy)
            );
            index = Some(new_index);
            info!("✅ Index HNSW baru berhasil dibuat");
        }
        
        // Load document map
        let document_map_path = data_dir.join("document_map.json");
        let document_map = if document_map_path.exists() {
            match fs::read_to_string(&document_map_path) {
                Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
                Err(_) => HashMap::new(),
            }
        } else {
            HashMap::new()
        };
        
        // Calculate next_id
        let next_id = document_map.keys().map(|&k| k + 1).max().unwrap_or(0);
        
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            index: Arc::new(Mutex::new(index)),
            document_map: Arc::new(Mutex::new(document_map)),
            next_id: Arc::new(Mutex::new(next_id)),
        })
    }
    
    pub fn preprocess_text(&self, text: &str) -> String {
        text.trim().replace("\n", " ")
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let model = self.model.lock().unwrap();
        let embeddings = model.encode(&[text])?;
        
        // Konversi embeddings ke Vec<f32>
        let embedding = embeddings[0].clone().into_iter().collect();
        
        Ok(embedding)
    }
    
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let model = self.model.lock().unwrap();
        let embeddings = model.encode(texts)?;
        
        // Konversi batch embeddings ke Vec<Vec<f32>>
        let results: Vec<Vec<f32>> = embeddings.into_iter()
            .map(|emb| emb.into_iter().collect())
            .collect();
            
        Ok(results)
    }
    
    pub fn add_document(&self, text: &str, doc_id_str: &str) -> Result<()> {
        // Parse doc_id atau gunakan next_id
        let doc_id = match doc_id_str.parse::<usize>() {
            Ok(id) => id,
            Err(_) => {
                let mut next_id = self.next_id.lock().unwrap();
                let id = *next_id;
                *next_id += 1;
                id
            }
        };
        
        // Encode dokumen
        let embedding = self.encode(text)?;
        
        // Tambahkan ke index
        let mut locked_index = self.index.lock().unwrap();
        if let Some(index) = locked_index.as_mut() {
            index.insert(&embedding, doc_id);
            
            // Tambahkan ke document map
            let mut document_map = self.document_map.lock().unwrap();
            document_map.insert(doc_id, text.to_string());
            
            // Simpan perubahan
            self.save_index()?;
            self.save_document_map()?;
        }
        
        Ok(())
    }
    
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<String>> {
        let embedding = self.encode(query)?;
        
        let locked_index = self.index.lock().unwrap();
        if let Some(index) = locked_index.as_ref() {
            // Check if index is empty
            if index.len() == 0 {
                return Ok(Vec::new());
            }
            
            // Lakukan pencarian
            let k = std::cmp::min(top_k, index.len());
            if k == 0 {
                return Ok(Vec::new());
            }
            
            let search_result = index.search(&embedding, k, SearchType::Default);
            
            // Ambil dokumen berdasarkan indeks
            let document_map = self.document_map.lock().unwrap();
            let mut documents = Vec::new();
            
            for (idx, _) in search_result {
                if let Some(doc) = document_map.get(&idx) {
                    documents.push(doc.clone());
                }
            }
            
            Ok(documents)
        } else {
            Ok(Vec::new())
        }
    }
    
    // Simpan index ke file
    pub fn save_index(&self) -> Result<()> {
        let locked_index = self.index.lock().unwrap();
        
        if let Some(index) = locked_index.as_ref() {
            let data_dir = Path::new("./db");
            if !data_dir.exists() {
                fs::create_dir_all(data_dir)?;
            }
            
            let index_path = data_dir.join("hnsw_index.json");
            let serialized = serde_json::to_string(&index)?;
            fs::write(&index_path, serialized)?;
            info!("✅ Index HNSW berhasil disimpan ke {:?}", index_path);
        }
        
        Ok(())
    }
    
    // Simpan document map ke file
    pub fn save_document_map(&self) -> Result<()> {
        let document_map = self.document_map.lock().unwrap();
        
        let data_dir = Path::new("./db");
        if !data_dir.exists() {
            fs::create_dir_all(data_dir)?;
        }
        
        let document_map_path = data_dir.join("document_map.json");
        let json = serde_json::to_string(&*document_map)?;
        fs::write(&document_map_path, json)?;
        
        info!("✅ Document map berhasil disimpan ke {:?}", document_map_path);
        
        Ok(())
    }
    
    // Hapus dokumen dari index
    pub fn delete_document(&self, doc_id_str: &str) -> Result<()> {
        let doc_id = doc_id_str.parse::<usize>()?;
        
        // Hapus dari document map
        let mut document_map = self.document_map.lock().unwrap();
        document_map.remove(&doc_id);
        
        self.save_document_map()?;
        
        // Note: HNSW tidak mendukung penghapusan langsung
        warn!("⚠️ Dokumen dihapus dari document map, tapi masih ada di HNSW index");
        warn!("⚠️ Untuk menghapus sepenuhnya, index perlu di-rebuild");
        
        Ok(())
    }
}

// Inisialisasi model embedding
pub fn init_model() -> Result<EmbeddingModel> {
    EmbeddingModel::new()
} 