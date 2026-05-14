//! Example demonstrating mistral.rs backend usage
//!
//! This example shows how to use mistral.rs as a local LLM backend for chat and embeddings.
//!
//! Two modes are supported:
//! 1. **Embedded mode**: Runs inference in-process using the mistralrs crate
//!    - Enable with: `cargo run --example mistral_rs_example --features mistral_rs`
//!    - Model is downloaded and loaded locally
//!    - Chat only (no streaming yet)
//!
//! 2. **Server mode**: Connects to a running mistral.rs HTTP server
//!    - Start server: `mistralrs serve -m Qwen/Qwen3-4B`
//!    - Run: `cargo run --example mistral_rs_example --features mistral_rs_server`
//!    - Full chat + streaming + embeddings support

use llm::builder::{LLMBackend, LLMBuilder};
use llm::chat::ChatMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== mistral.rs Backend Example ===\n");

    // Example 1: Embedded mode (requires `mistral_rs` feature)
    // Uses a HuggingFace model ID (contains '/') to trigger embedded mode
    println!("1. Embedded Mode Example:");
    #[cfg(feature = "mistral_rs")]
    {
        let llm = LLMBuilder::new()
            .backend(LLMBackend::MistralRs)
            .model("Qwen/Qwen3-4B") // HF model ID triggers embedded mode
            .quantization("q4")     // 4-bit quantization for lower memory
            .max_tokens(256)
            .temperature(0.7)
            .build()?;

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
                println!("Error: {}\n", e);
            }
        }
    }
    #[cfg(not(feature = "mistral_rs"))]
    {
        println!("   Skipped (enable 'mistral_rs' feature for embedded mode)\n");
    }

    // Example 2: Server mode (connects to running mistral.rs server)
    println!("2. Server Mode Example:");
    {
        let llm = LLMBuilder::new()
            .backend(LLMBackend::MistralRs)
            .base_url("http://localhost:1234")
            .model("default") // Simple name triggers server mode
            .max_tokens(256)
            .temperature(0.7)
            .build()?;

        let messages = vec![
            ChatMessage::user()
                .content("Tell me a short joke about programming.")
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
    }

    // Example 3: Streaming chat (server mode only)
    println!("3. Streaming Chat Example (server mode):");
    {
        use futures::StreamExt;

        let llm = LLMBuilder::new()
            .backend(LLMBackend::MistralRs)
            .base_url("http://localhost:1234")
            .model("default")
            .build()?;

        let messages = vec![
            ChatMessage::user()
                .content("Count from 1 to 5.")
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
                println!("Error: {}\n", e);
            }
        }
    }

    // Example 4: Embeddings (server mode only)
    println!("4. Embedding Example (server mode):");
    {
        let llm = LLMBuilder::new()
            .backend(LLMBackend::MistralRs)
            .base_url("http://localhost:1234")
            .model("default")
            .build()?;

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
                println!("Error: {}", e);
            }
        }
    }

    println!("\n=== Example Complete ===");
    println!("\nUsage:");
    println!("  Embedded: cargo run --example mistral_rs_example --features mistral_rs");
    println!("  Server:   mistralrs serve -m Qwen/Qwen3-4B && cargo run --example mistral_rs_example");

    Ok(())
}
