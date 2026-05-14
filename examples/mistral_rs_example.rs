//! Example demonstrating mistral.rs backend usage
//!
//! This example shows how to use mistral.rs as a local LLM backend for chat and embeddings.
//!
//! Prerequisites:
//! 1. Install mistral.rs: https://github.com/EricLBuehler/mistral.rs
//! 2. Start mistral.rs server: `mistralrs serve -m Qwen/Qwen3-4B`
//! 3. Run this example: `cargo run --example mistral_rs_example --features mistral_rs`

use llm::builder::{LLMBackend, LLMBuilder};
use llm::chat::ChatMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== mistral.rs Backend Example ===\n");

    // Create a mistral.rs backend using the builder
    // By default, connects to http://localhost:1234
    let llm = LLMBuilder::new()
        .backend(LLMBackend::MistralRs)
        .base_url("http://localhost:1234")
        .model("default")
        .max_tokens(256)
        .temperature(0.7)
        .build()?;

    // Example 1: Simple chat
    println!("1. Simple Chat Example:");
    let messages = vec![
        ChatMessage::user()
            .content("What is the capital of France?")
            .build(),
    ];

    match llm.chat(&messages).await {
        Ok(response) => {
            println!("Response: {}\n", response.text().unwrap_or_default());
        }
        Err(e) => {
            println!("Error (is mistral.rs running?): {}\n", e);
        }
    }

    // Example 2: Streaming chat
    println!("2. Streaming Chat Example:");
    use futures::StreamExt;

    let messages = vec![
        ChatMessage::user()
            .content("Tell me a short joke about programming.")
            .build(),
    ];

    match llm.chat_stream(&messages).await {
        Ok(mut stream) => {
            print!("Response: ");
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(text) => print!("{}", text),
                    Err(e) => eprintln!("\nError: {}", e),
                }
            }
            println!("\n");
        }
        Err(e) => {
            println!("Error (is mistral.rs running?): {}\n", e);
        }
    }

    // Example 3: Embeddings
    println!("3. Embedding Example:");
    let texts = vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Machine learning is a subset of artificial intelligence".to_string(),
    ];

    match llm.embed(texts.clone()).await {
        Ok(embeddings) => {
            for (i, embedding) in embeddings.iter().enumerate() {
                println!(
                    "Text {}: '{}' -> Embedding dimension: {}",
                    i + 1,
                    texts[i],
                    embedding.len()
                );
            }
        }
        Err(e) => {
            println!("Error (is mistral.rs running with an embedding model?): {}", e);
        }
    }

    println!("\n=== Example Complete ===");
    println!("\nNote: Make sure mistral.rs is running with `mistralrs serve -m <model>`");

    Ok(())
}
