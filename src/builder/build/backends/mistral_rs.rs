#[cfg(feature = "mistral_rs")]
use crate::backends::mistral_rs::{MistralRs, MistralRsMode, MistralRsQuantization};

use crate::{builder::state::BuilderState, chat::Tool, error::LLMError, LLMProvider};
use secrecy::ExposeSecret;

#[cfg(feature = "mistral_rs")]
pub(super) fn build_mistral_rs(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    // Check if embedded mode is requested via model_id format
    // If model looks like a HF model ID (contains /), use embedded mode
    let model = state.model.as_deref().unwrap_or("default");
    let use_embedded = model.contains('/') || model.contains('\\');

    if use_embedded {
        // Embedded mode - load model in-process
        let quantization = state
            .quantization
            .as_ref()
            .map(|q| match q.as_str() {
                "q4" | "4" => MistralRsQuantization::Q4,
                "q8" | "8" => MistralRsQuantization::Q8,
                _ => MistralRsQuantization::None,
            })
            .unwrap_or_default();

        // Use tokio runtime to build the async model
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| LLMError::ProviderError(format!("Failed to create runtime: {}", e)))?;

        let client = rt.block_on(MistralRs::new_embedded(
            model,
            quantization,
            state.max_tokens,
            state.temperature,
            state.system.take(),
            state.top_p,
            state.top_k,
            state.json_schema.take(),
            tools,
        ))?;

        Ok(Box::new(client))
    } else {
        // Server mode - connect to running server
        let base_url = state
            .base_url
            .take()
            .unwrap_or_else(|| "http://localhost:1234".to_string());

        let client = MistralRs::new_server(
            base_url,
            state.api_key.take().map(|k| k.expose_secret().clone()),
            Some(model.to_string()),
            state.max_tokens,
            state.temperature,
            state.timeout_seconds,
            state.system.take(),
            state.top_p,
            state.top_k,
            state.json_schema.take(),
            tools,
        );

        Ok(Box::new(client))
    }
}

#[cfg(not(feature = "mistral_rs"))]
pub(super) fn build_mistral_rs(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    // Fallback to server mode when embedded feature is not enabled
    let base_url = state
        .base_url
        .take()
        .unwrap_or_else(|| "http://localhost:1234".to_string());
    let model = state.model.take().unwrap_or_else(|| "default".to_string());

    let client = crate::backends::mistral_rs::MistralRs::new_server(
        base_url,
        state.api_key.take().map(|k| k.expose_secret().clone()),
        Some(model),
        state.max_tokens,
        state.temperature,
        state.timeout_seconds,
        state.system.take(),
        state.top_p,
        state.top_k,
        state.json_schema.take(),
        tools,
    );

    Ok(Box::new(client))
}
