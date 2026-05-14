//! mistral.rs API client implementation for chat and embedding functionality.
//!
//! This module provides integration with mistral.rs local LLM server through its OpenAI-compatible API.

use std::pin::Pin;
use std::sync::Arc;

use crate::{
    builder::LLMBackend,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StreamChunk,
        StructuredOutputFormat, Tool,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRawEntry, ModelListRequest, ModelListResponse, ModelsProvider},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    FunctionCall, ToolCall,
};
use async_trait::async_trait;
use base64::{self, Engine};
use chrono::{DateTime, Utc};
use futures::Stream;
use reqwest::Client;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for the mistral.rs client.
#[derive(Debug)]
pub struct MistralRsConfig {
    /// Base URL for the mistral.rs API.
    pub base_url: String,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Model identifier.
    pub model: String,
    /// Maximum tokens to generate in responses.
    pub max_tokens: Option<u32>,
    /// Sampling temperature for response randomness.
    pub temperature: Option<f32>,
    /// System prompt to guide model behavior.
    pub system: Option<String>,
    /// Request timeout in seconds.
    pub timeout_seconds: Option<u64>,
    /// Top-p (nucleus) sampling parameter.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter.
    pub top_k: Option<u32>,
    /// JSON schema for structured output.
    pub json_schema: Option<StructuredOutputFormat>,
    /// Available tools for the model to use.
    pub tools: Option<Vec<Tool>>,
}

/// Client for interacting with mistral.rs's OpenAI-compatible API.
///
/// Provides methods for chat and completion requests using mistral.rs's models.
///
/// The client uses `Arc` internally for configuration, making cloning cheap.
#[derive(Debug, Clone)]
pub struct MistralRs {
    /// Shared configuration wrapped in Arc for cheap cloning.
    pub config: Arc<MistralRsConfig>,
    /// HTTP client for making requests.
    pub client: Client,
}

/// Request payload for OpenAI-compatible chat API endpoint.
#[derive(Serialize)]
struct MistralRsChatRequest<'a> {
    model: String,
    messages: Vec<MistralRsChatMessage<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MistralRsTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
}

/// Individual message in a chat conversation.
#[derive(Serialize)]
struct MistralRsChatMessage<'a> {
    role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<MistralRsMessageContent<'a>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum MistralRsMessageContent<'a> {
    Text(&'a str),
    Multimodal(Vec<MistralRsContentPart>),
}

#[derive(Serialize)]
struct MistralRsContentPart {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<MistralRsImageUrl>,
}

#[derive(Serialize)]
struct MistralRsImageUrl {
    url: String,
}

impl<'a> From<&'a ChatMessage> for MistralRsChatMessage<'a> {
    fn from(msg: &'a ChatMessage) -> Self {
        let role = match msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };

        let content = match &msg.message_type {
            MessageType::Text => Some(MistralRsMessageContent::Text(&msg.content)),
            MessageType::Image((_mime, data)) => {
                let base64_data = base64::engine::general_purpose::STANDARD.encode(data);
                let data_url = format!("data:image/jpeg;base64,{}", base64_data);
                Some(MistralRsMessageContent::Multimodal(vec![MistralRsContentPart {
                    content_type: "image_url".to_string(),
                    text: None,
                    image_url: Some(MistralRsImageUrl { url: data_url }),
                }]))
            }
            MessageType::ImageURL(url) => Some(MistralRsMessageContent::Multimodal(vec![
                MistralRsContentPart {
                    content_type: "image_url".to_string(),
                    text: None,
                    image_url: Some(MistralRsImageUrl { url: url.clone() }),
                },
            ])),
            _ => Some(MistralRsMessageContent::Text(&msg.content)),
        };

        Self { role, content }
    }
}

/// Response from mistral.rs API endpoints.
#[derive(Deserialize, Debug)]
struct MistralRsChatResponse {
    id: Option<String>,
    choices: Vec<MistralRsChoice>,
    usage: Option<MistralRsUsage>,
}

#[derive(Deserialize, Debug)]
struct MistralRsChoice {
    message: MistralRsResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct MistralRsResponseMessage {
    content: Option<String>,
    #[serde(rename = "tool_calls")]
    tool_calls: Option<Vec<MistralRsToolCall>>,
}

#[derive(Deserialize, Debug)]
struct MistralRsToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: MistralRsFunctionCall,
}

#[derive(Deserialize, Debug)]
struct MistralRsFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Deserialize, Debug)]
struct MistralRsUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

impl std::fmt::Display for MistralRsChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(choice) = self.choices.first() {
            if let Some(content) = &choice.message.content {
                return write!(f, "{}", content);
            }
        }
        Ok(())
    }
}

impl ChatResponse for MistralRsChatResponse {
    fn text(&self) -> Option<String> {
        self.choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.to_string())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices.first().and_then(|c| {
            c.message.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone(),
                        call_type: tc.call_type.clone(),
                        function: FunctionCall {
                            name: tc.function.name.clone(),
                            arguments: serde_json::to_string(&tc.function.arguments)
                                .unwrap_or_default(),
                        },
                    })
                    .collect()
            })
        })
    }

    fn usage(&self) -> Option<crate::chat::Usage> {
        self.usage.as_ref().map(|u| crate::chat::Usage {
            prompt_tokens: u.prompt_tokens.unwrap_or(0),
            completion_tokens: u.completion_tokens.unwrap_or(0),
            total_tokens: u.total_tokens.unwrap_or(0),
            completion_tokens_details: None,
            prompt_tokens_details: None,
        })
    }
}

/// Tool definition for mistral.rs
#[derive(Serialize, Debug)]
struct MistralRsTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: MistralRsFunctionTool,
}

#[derive(Serialize, Debug)]
struct MistralRsFunctionTool {
    name: String,
    description: String,
    parameters: Value,
}

impl From<&crate::chat::Tool> for MistralRsTool {
    fn from(tool: &crate::chat::Tool) -> Self {
        MistralRsTool {
            tool_type: "function".to_owned(),
            function: MistralRsFunctionTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
            },
        }
    }
}

/// Request payload for embedding API endpoint.
#[derive(Serialize)]
struct MistralRsEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct MistralRsEmbeddingResponse {
    data: Vec<MistralRsEmbeddingData>,
}

#[derive(Deserialize, Debug)]
struct MistralRsEmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

/// Request payload for streaming chat.
#[derive(Serialize)]
struct MistralRsChatStreamRequest<'a> {
    model: String,
    messages: Vec<MistralRsChatMessage<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MistralRsTool>>,
}

/// Streaming response chunk
#[derive(Deserialize, Debug)]
struct MistralRsStreamChunk {
    id: Option<String>,
    choices: Vec<MistralRsStreamChoice>,
}

#[derive(Deserialize, Debug)]
struct MistralRsStreamChoice {
    delta: MistralRsStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct MistralRsStreamDelta {
    content: Option<String>,
    #[serde(rename = "tool_calls")]
    tool_calls: Option<Vec<MistralRsStreamToolCall>>,
}

#[derive(Deserialize, Debug)]
struct MistralRsStreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    #[serde(rename = "type")]
    call_type: Option<String>,
    function: Option<MistralRsStreamFunction>,
}

#[derive(Deserialize, Debug)]
struct MistralRsStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

/// Model listing response
#[derive(Deserialize, Debug)]
struct MistralRsModelListResponse {
    data: Vec<MistralRsModelData>,
}

#[derive(Deserialize, Debug, Clone)]
struct MistralRsModelData {
    id: String,
    created: Option<i64>,
    #[serde(flatten)]
    extra: Value,
}

impl ModelListRawEntry for MistralRsModelData {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created
            .map(|ts| DateTime::from_timestamp(ts, 0).unwrap_or(DateTime::<Utc>::UNIX_EPOCH))
            .unwrap_or(DateTime::<Utc>::UNIX_EPOCH)
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

impl ModelListResponse for MistralRsModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.data.iter().map(|m| m.id.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.data
            .clone()
            .into_iter()
            .map(|e| Box::new(e) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::MistralRs
    }
}

impl MistralRs {
    /// Creates a new mistral.rs client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the mistral.rs server (e.g., "http://localhost:1234")
    /// * `api_key` - Optional API key for authentication
    /// * `model` - Model name to use
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt
    /// * `top_p` - Top-p sampling parameter
    /// * `top_k` - Top-k sampling parameter
    /// * `json_schema` - JSON schema for structured output
    /// * `tools` - Function tools that the model can use
    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        json_schema: Option<StructuredOutputFormat>,
        tools: Option<Vec<Tool>>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self::with_client(
            builder.build().expect("Failed to build reqwest Client"),
            base_url,
            api_key,
            model,
            max_tokens,
            temperature,
            timeout_seconds,
            system,
            top_p,
            top_k,
            json_schema,
            tools,
        )
    }

    /// Creates a new mistral.rs client with a custom HTTP client.
    #[allow(clippy::too_many_arguments)]
    pub fn with_client(
        client: Client,
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        json_schema: Option<StructuredOutputFormat>,
        tools: Option<Vec<Tool>>,
    ) -> Self {
        Self {
            config: Arc::new(MistralRsConfig {
                base_url: base_url.into(),
                api_key,
                model: model.unwrap_or("default".to_string()),
                temperature,
                max_tokens,
                timeout_seconds,
                system,
                top_p,
                top_k,
                json_schema,
                tools,
            }),
            client,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    pub fn api_key(&self) -> Option<&str> {
        self.config.api_key.as_deref()
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    pub fn max_tokens(&self) -> Option<u32> {
        self.config.max_tokens
    }

    pub fn temperature(&self) -> Option<f32> {
        self.config.temperature
    }

    pub fn timeout_seconds(&self) -> Option<u64> {
        self.config.timeout_seconds
    }

    pub fn system(&self) -> Option<&str> {
        self.config.system.as_deref()
    }

    pub fn top_p(&self) -> Option<f32> {
        self.config.top_p
    }

    pub fn top_k(&self) -> Option<u32> {
        self.config.top_k
    }

    pub fn json_schema(&self) -> Option<&StructuredOutputFormat> {
        self.config.json_schema.as_ref()
    }

    pub fn tools(&self) -> Option<&[Tool]> {
        self.config.tools.as_deref()
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    fn make_chat_request<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        tools: Option<&'a [Tool]>,
        stream: bool,
    ) -> MistralRsChatRequest<'a> {
        let mut chat_messages: Vec<MistralRsChatMessage> =
            messages.iter().map(MistralRsChatMessage::from).collect();

        if let Some(system) = &self.config.system {
            chat_messages.insert(
                0,
                MistralRsChatMessage {
                    role: "system",
                    content: Some(MistralRsMessageContent::Text(system)),
                },
            );
        }

        let mistral_rs_tools = tools.map(|t| t.iter().map(MistralRsTool::from).collect());

        let response_format = if let Some(schema) = &self.config.json_schema {
            schema.schema.as_ref().map(|s| {
                serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "schema": s
                    }
                })
            })
        } else {
            None
        };

        MistralRsChatRequest {
            model: self.config.model.clone(),
            messages: chat_messages,
            stream,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            tools: mistral_rs_tools,
            response_format,
        }
    }
}

const AUDIO_UNSUPPORTED: &str = "Audio messages are not supported by mistral.rs chat";

#[async_trait]
impl ChatProvider for MistralRs {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        crate::chat::ensure_no_audio(messages, AUDIO_UNSUPPORTED)?;
        if self.config.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }

        let req_body = self.make_chat_request(messages, tools, false);

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("mistral.rs request payload (tools): {}", json);
            }
        }

        let url = format!("{}/v1/chat/completions", self.config.base_url);

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(api_key) = &self.config.api_key {
            request = request.bearer_auth(api_key);
        }

        if let Some(timeout) = self.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("mistral.rs HTTP status (tools): {}", resp.status());

        let resp = resp.error_for_status()?;
        let json_resp = resp.json::<MistralRsChatResponse>().await?;

        Ok(Box::new(json_resp))
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        crate::chat::ensure_no_audio(messages, AUDIO_UNSUPPORTED)?;
        let req_body = self.make_chat_request(messages, None, true);

        let url = format!("{}/v1/chat/completions", self.config.base_url);
        let mut request = self.client.post(&url).json(&req_body);

        if let Some(api_key) = &self.config.api_key {
            request = request.bearer_auth(api_key.as_str());
        }

        if let Some(timeout) = self.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;
        log::debug!("mistral.rs HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        Ok(crate::chat::create_sse_stream(resp, parse_mistral_rs_sse))
    }
}

#[async_trait]
impl CompletionProvider for MistralRs {
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        if self.config.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }
        let url = format!("{}/v1/completions", self.config.base_url);

        let completion_req = serde_json::json!({
            "model": self.config.model,
            "prompt": req.prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        });

        let mut request = self.client.post(&url).json(&completion_req);

        if let Some(api_key) = &self.config.api_key {
            request = request.bearer_auth(api_key);
        }

        if let Some(timeout) = self.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?.error_for_status()?;
        let json_resp: Value = resp.json().await?;

        if let Some(text) = json_resp["choices"][0]["text"].as_str() {
            Ok(CompletionResponse {
                text: text.to_string(),
            })
        } else {
            Err(LLMError::ProviderError(
                "No answer returned by mistral.rs".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EmbeddingProvider for MistralRs {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.config.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }
        let url = format!("{}/v1/embeddings", self.config.base_url);

        let body = MistralRsEmbeddingRequest {
            model: self.config.model.clone(),
            input,
        };

        let mut request = self.client.post(&url).json(&body);

        if let Some(api_key) = &self.config.api_key {
            request = request.bearer_auth(api_key);
        }

        if let Some(timeout) = self.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?.error_for_status()?;
        let json_resp: MistralRsEmbeddingResponse = resp.json().await?;

        let mut embeddings = vec![vec![]; json_resp.data.len()];
        for data in json_resp.data {
            embeddings[data.index] = data.embedding;
        }

        Ok(embeddings)
    }
}

#[async_trait]
impl SpeechToTextProvider for MistralRs {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "mistral.rs does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for MistralRs {
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError(
            "mistral.rs does not implement text to speech endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for MistralRs {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        if self.config.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }

        let url = format!("{}/v1/models", self.config.base_url);

        let mut request = self.client.get(&url);

        if let Some(api_key) = &self.config.api_key {
            request = request.bearer_auth(api_key);
        }

        if let Some(timeout) = self.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?.error_for_status()?;
        let result: MistralRsModelListResponse = resp.json().await?;
        Ok(Box::new(result))
    }
}

impl crate::LLMProvider for MistralRs {
    fn tools(&self) -> Option<&[Tool]> {
        self.config.tools.as_deref()
    }
}

/// Parses a Server-Sent Events (SSE) chunk from mistral.rs's streaming API.
///
/// # Arguments
///
/// * `chunk` - The raw SSE chunk text
///
/// # Returns
///
/// * `Ok(Some(String))` - Content token if found
/// * `Ok(None)` - If chunk should be skipped (e.g., ping, done signal)
/// * `Err(LLMError)` - If parsing fails
fn parse_mistral_rs_sse(chunk: &str) -> Result<Option<String>, LLMError> {
    let mut collected_content = String::new();

    for line in chunk.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }

            match serde_json::from_str::<MistralRsStreamChunk>(data) {
                Ok(response) => {
                    for choice in &response.choices {
                        if let Some(content) = &choice.delta.content {
                            collected_content.push_str(content);
                        }
                    }
                }
                Err(e) => return Err(LLMError::JsonError(e.to_string())),
            }
        }
    }

    if collected_content.is_empty() {
        Ok(None)
    } else {
        Ok(Some(collected_content))
    }
}
