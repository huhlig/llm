#[cfg(feature = "mistral_rs")]
use crate::backends::mistral_rs::MistralRs;

use crate::{builder::state::BuilderState, chat::Tool, error::LLMError, LLMProvider};
use secrecy::ExposeSecret;

#[cfg(feature = "mistral_rs")]
pub(super) fn build_mistral_rs(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let base_url = state
        .base_url
        .take()
        .unwrap_or_else(|| "http://localhost:1234".to_string());
    let model = state.model.take().unwrap_or_else(|| "default".to_string());

    let client = MistralRs::new(
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

#[cfg(not(feature = "mistral_rs"))]
pub(super) fn build_mistral_rs(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "mistral_rs feature is not enabled".to_string(),
    ))
}
