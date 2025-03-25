use anyhow::{Context, Result};
use csv::Reader;
use log::error;
use pdf_extract::extract_text;
use std::io::Cursor;
use std::io::Read;

// Preprocessing teks
pub fn preprocess_text(text: &str) -> String {
    text.trim().replace("\n", " ")
}

// Ekstrak teks dari file PDF
pub fn extract_pdf_text(data: &[u8]) -> Result<String> {
    let text = extract_text(Cursor::new(data))
        .context("Gagal mengekstrak teks dari PDF")?;
    
    Ok(text)
}

// Ekstrak teks dari file CSV
pub fn extract_csv_text(data: &[u8]) -> Result<String> {
    let cursor = Cursor::new(data);
    let mut rdr = Reader::from_reader(cursor);
    let mut text = String::new();
    
    for result in rdr.records() {
        let record = result.context("Gagal membaca record CSV")?;
        for field in record.iter() {
            text.push_str(field);
            text.push_str(" ");
        }
        text.push_str("\n");
    }
    
    Ok(text)
}

// Ekstrak teks dari file TXT
pub fn extract_txt_text(data: &[u8]) -> Result<String> {
    let mut text = String::new();
    let mut cursor = Cursor::new(data);
    cursor.read_to_string(&mut text)
        .context("Gagal membaca file teks")?;
    
    Ok(text)
}

// Helper untuk mendapatkan extension dari filename
pub fn get_file_extension(filename: &str) -> Option<&str> {
    let parts: Vec<&str> = filename.split('.').collect();
    if parts.len() > 1 {
        Some(parts.last().unwrap())
    } else {
        None
    }
} 