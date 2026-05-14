//! mistral.rs backend - embedded local LLM inference with optional server mode.
//!
//! This module provides two modes:
//! - **Embedded**: In-process inference using the `mistralrs` crate directly
//! - **Server**: HTTP connection to a running mistral.rs server (OpenAI-compatible API)
//!
//! Embedded mode requires the `mistral_rs` feature and downloads/models the model locally.
//! Server mode requires the `mistral_rs_server` feature and connects to a running server.

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
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Quantization bits for embedded mode
#[derive(Debug, Clone, Copy, Default)]
pub enum MistralRsQuantization {
    /// 4-bit quantization (Q4)
    Q4,
    /// 8-bit quantization (Q8)
    Q8,
    /// No quantization (FP16)
    #[default]
    None,
}

/// Mode for mistral.rs backend
#[derive(Debug, Clone)]
pub enum MistralRsMode {
    /// Embedded mode - runs inference in-process using the mistralrs crate
    Embedded {
        /// HuggingFace model ID or local path (e.g., "Qwen/Qwen3-4B")
        model_id: String,
        /// Quantization level
        quantization: MistralRsQuantization,
        /// Enable PagedAttention for better memory efficiency
        paged_attention: bool,
    },
    /// Server mode - connects to a running mistral.rs HTTP server
    Server {
        /// Base URL of the mistral.rs server (e.g., "http://localhost:1234")
        base_url: String,
        /// Optional API key for authentication
        api_key: Option<String>,
        /// Model name as known to the server
        model: String,
    },
}

impl Default for MistralRsMode {
    fn default() -> Self {
        MistralRsMode::Server {
            base_url: "http://localhost:1234".to_string(),
            api_key: None,
            model: "default".to_string(),
        }
    }
}

/// Configuration for the mistral.rs client.
#[derive(Debug)]
pub struct MistralRsConfig {
    /// Mode: embedded or server
    pub mode: MistralRsMode,
    /// Maximum tokens to generate in responses.
    pub max_tokens: Option<u32>,
    /// Sampling temperature for response randomness.
    pub temperature: Option<f32>,
    /// System prompt to guide model behavior.
    pub system: Option<String>,
    /// Request timeout in seconds (server mode only).
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

/// Client for mistral.rs - supports both embedded and server modes.
pub struct MistralRs {
    /// Shared configuration wrapped in Arc for cheap cloning.
    pub config: Arc<MistralRsConfig>,
    /// HTTP client for server mode.
    pub client: reqwest::Client,
    /// Embedded model handle (only valid in embedded mode).
    #[cfg(feature = "mistral_rs")]
    pub embedded_model: Option<std::sync::Arc<mistralrs::Model>>,
}

// Server mode types
#[derive(Serialize)]
struct ServerChatRequest<'a> {
    model: String,
    messages: Vec<ServerChatMessage<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ServerTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
}

#[derive(Serialize)]
struct ServerChatMessage<'a> {
    role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<ServerMessageContent<'a>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ServerMessageContent<'a> {
    Text(&'a str),
    Multimodal(Vec<ServerContentPart>),
}

#[derive(Serialize)]
struct ServerContentPart {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ServerImageUrl>,
}

#[derive(Serialize)]
struct ServerImageUrl {
    url: String,
}

impl<'a> From<&'a ChatMessage> for ServerChatMessage<'a> {
    fn from(msg: &'a ChatMessage) -> Self {
        let role = match msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };

        let content = match &msg.message_type {
            MessageType::Text => Some(ServerMessageContent::Text(&msg.content)),
            MessageType::Image((_mime, data)) => {
                let base64_data = base64::engine::general_purpose::STANDARD.encode(data);
                Some(ServerMessageContent::Multimodal(vec![ServerContentPart {
                    content_type: "image_url".to_string(),
                    text: None,
                    image_url: Some(ServerImageUrl {
                        url: format!("data:image/jpeg;base64,{}", base64_data),
                    }),
                }]))
            }
            MessageType::ImageURL(url) => Some(ServerMessageContent::Multimodal(vec![
                ServerContentPart {
                    content_type: "image_url".to_string(),
                    text: None,
                    image_url: Some(ServerImageUrl { url: url.clone() }),
                },
            ])),
            _ => Some(ServerMessageContent::Text(&msg.content)),
        };

        Self { role, content }
    }
}

#[derive(Deserialize, Debug)]
struct ServerChatResponse {
    id: Option<String>,
    choices: Vec<ServerChoice>,
    usage: Option<ServerUsage>,
}

#[derive(Deserialize, Debug)]
struct ServerChoice {
    message: ServerResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ServerResponseMessage {
    content: Option<String>,
    #[serde(rename = "tool_calls")]
    tool_calls: Option<Vec<ServerToolCall>>,
}

#[derive(Deserialize, Debug)]
struct ServerToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ServerFunctionCall,
}

#[derive(Deserialize, Debug)]
struct ServerFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Deserialize, Debug)]
struct ServerUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

impl std::fmt::Display for ServerChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(choice) = self.choices.first() {
            if let Some(content) = &choice.message.content {
                return write!(f, "{}", content);
            }
        }
        Ok(())
    }
}

impl ChatResponse for ServerChatResponse {
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

#[derive(Serialize, Debug)]
struct ServerTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ServerFunctionTool,
}

#[derive(Serialize, Debug)]
struct ServerFunctionTool {
    name: String,
    description: String,
    parameters: Value,
}

impl From<&crate::chat::Tool> for ServerTool {
    fn from(tool: &crate::chat::Tool) -> Self {
        ServerTool {
            tool_type: "function".to_owned(),
            function: ServerFunctionTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
            },
        }
    }
}

#[derive(Serialize)]
struct ServerEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct ServerEmbeddingResponse {
    data: Vec<ServerEmbeddingData>,
}

#[derive(Deserialize, Debug)]
struct ServerEmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct ServerStreamChunk {
    id: Option<String>,
    choices: Vec<ServerStreamChoice>,
}

#[derive(Deserialize, Debug)]
struct ServerStreamChoice {
    delta: ServerStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ServerStreamDelta {
    content: Option<String>,
    #[serde(rename = "tool_calls")]
    tool_calls: Option<Vec<ServerStreamToolCall>>,
}

#[derive(Deserialize, Debug)]
struct ServerStreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    #[serde(rename = "type")]
    call_type: Option<String>,
    function: Option<ServerStreamFunction>,
}

#[derive(Deserialize, Debug)]
struct ServerStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
struct ServerModelData {
    id: String,
    created: Option<i64>,
    #[serde(flatten)]
    extra: Value,
}

impl ModelListRawEntry for ServerModelData {
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

#[derive(Deserialize, Debug)]
struct ServerModelListResponse {
    data: Vec<ServerModelData>,
}

impl ModelListResponse for ServerModelListResponse {
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
    /// Creates a new mistral.rs client with embedded mode.
    #[cfg(feature = "mistral_rs")]
    pub async fn new_embedded(
        model_id: impl Into<String>,
        quantization: MistralRsQuantization,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        json_schema: Option<StructuredOutputFormat>,
        tools: Option<Vec<Tool>>,
    ) -> Result<Self, LLMError> {
        use mistralrs::{IsqBits, ModelBuilder, PagedAttentionMetaBuilder};

        let model_id = model_id.into();

        let mut builder = ModelBuilder::new(&model_id).with_auto_isq(match quantization {
            MistralRsQuantization::Q4 => IsqBits::Four,
            MistralRsQuantization::Q8 => IsqBits::Eight,
            MistralRsQuantization::None => IsqBits::Two, // Use lowest quantization as fallback
        });

        builder = builder.with_paged_attn(PagedAttentionMetaBuilder::default().build().map_err(|e| {
            LLMError::ProviderError(format!("Failed to build paged attention: {}", e))
        })?);

        let model = builder
            .build()
            .await
            .map_err(|e| LLMError::ProviderError(format!("Failed to load model: {}", e)))?;

        Ok(Self {
            config: Arc::new(MistralRsConfig {
                mode: MistralRsMode::Embedded {
                    model_id,
                    quantization,
                    paged_attention: true,
                },
                max_tokens,
                temperature,
                system,
                timeout_seconds: None,
                top_p,
                top_k,
                json_schema,
                tools,
            }),
            client: reqwest::Client::new(),
            embedded_model: Some(Arc::new(model)),
        })
    }

    /// Creates a new mistral.rs client with server mode.
    pub fn new_server(
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
        let mut builder = reqwest::Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            config: Arc::new(MistralRsConfig {
                mode: MistralRsMode::Server {
                    base_url: base_url.into(),
                    api_key,
                    model: model.unwrap_or("default".to_string()),
                },
                max_tokens,
                temperature,
                system,
                timeout_seconds,
                top_p,
                top_k,
                json_schema,
                tools,
            }),
            client: builder
                .build()
                .expect("Failed to build reqwest Client"),
            #[cfg(feature = "mistral_rs")]
            embedded_model: None,
        }
    }

    fn is_embedded(&self) -> bool {
        matches!(self.config.mode, MistralRsMode::Embedded { .. })
    }

    fn get_server_config(&self) -> Option<(&str, &str, Option<&str>)> {
        match &self.config.mode {
            MistralRsMode::Server {
                base_url,
                model,
                api_key,
            } => Some((base_url, model, api_key.as_deref())),
            _ => None,
        }
    }

    fn make_server_chat_request<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        tools: Option<&'a [Tool]>,
        stream: bool,
    ) -> Option<ServerChatRequest<'a>> {
        let (base_url, model, _) = self.get_server_config()?;
        if base_url.is_empty() {
            return None;
        }

        let mut chat_messages: Vec<ServerChatMessage> =
            messages.iter().map(ServerChatMessage::from).collect();

        if let Some(system) = &self.config.system {
            chat_messages.insert(
                0,
                ServerChatMessage {
                    role: "system",
                    content: Some(ServerMessageContent::Text(system)),
                },
            );
        }

        let server_tools = tools.map(|t| t.iter().map(ServerTool::from).collect());

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

        Some(ServerChatRequest {
            model: model.to_string(),
            messages: chat_messages,
            stream,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            tools: server_tools,
            response_format,
        })
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

        if self.is_embedded() {
            #[cfg(feature = "mistral_rs")]
            {
                if let Some(model) = &self.embedded_model {
                    use mistralrs::{RequestBuilder, TextMessageRole, TextMessages};

                    let mut text_messages = TextMessages::new();

                    if let Some(system) = &self.config.system {
                        text_messages = text_messages.add_message(
                            TextMessageRole::System,
                            system.as_str(),
                        );
                    }

                    for msg in messages {
                        let role = match msg.role {
                            ChatRole::User => TextMessageRole::User,
                            ChatRole::Assistant => TextMessageRole::Assistant,
                        };
                        text_messages = text_messages.add_message(role, msg.content.as_str());
                    }

                    let response = model
                        .send_chat_request(text_messages)
                        .await
                        .map_err(|e| LLMError::ProviderError(format!("Chat error: {}", e)))?;

                    return Ok(Box::new(EmbeddedChatResponse(response)));
                }
            }
            return Err(LLMError::ProviderError(
                "Embedded model not loaded".to_string(),
            ));
        }

        // Server mode
        let req_body = self
            .make_server_chat_request(messages, tools, false)
            .ok_or_else(|| LLMError::InvalidRequest("Missing server configuration".to_string()))?;

        let (base_url, _, api_key) = self.get_server_config().unwrap();
        let url = format!("{}/v1/chat/completions", base_url);

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }

        let resp = request.send().await?;
        log::debug!("mistral.rs HTTP status (tools): {}", resp.status());

        let resp = resp.error_for_status()?;
        let json_resp = resp.json::<ServerChatResponse>().await?;

        Ok(Box::new(json_resp))
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        crate::chat::ensure_no_audio(messages, AUDIO_UNSUPPORTED)?;

        if self.is_embedded() {
            return Err(LLMError::ProviderError(
                "Streaming not yet supported in embedded mode. Use server mode for streaming.".into(),
            ));
        }

        // Server mode
        let req_body = self
            .make_server_chat_request(messages, None, true)
            .ok_or_else(|| LLMError::InvalidRequest("Missing server configuration".to_string()))?;

        let (base_url, _, api_key) = self.get_server_config().unwrap();
        let url = format!("{}/v1/chat/completions", base_url);
        let mut request = self.client.post(&url).json(&req_body);

        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }

        let resp = request.send().await?;
        log::debug!("mistral.rs HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        Ok(crate::chat::create_sse_stream(resp, parse_server_sse))
    }
}

#[cfg(feature = "mistral_rs")]
struct EmbeddedChatResponse(mistralrs::ChatCompletionResponse);

#[cfg(feature = "mistral_rs")]
impl std::fmt::Debug for EmbeddedChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedChatResponse").finish()
    }
}

#[cfg(feature = "mistral_rs")]
impl std::fmt::Display for EmbeddedChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(content) = self.0.choices.first().and_then(|c| c.message.content.as_ref()) {
            write!(f, "{}", content)
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "mistral_rs")]
impl ChatResponse for EmbeddedChatResponse {
    fn text(&self) -> Option<String> {
        self.0
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.to_string())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.0.choices.first().and_then(|c| {
            c.message.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone(),
                        call_type: "function".to_string(),
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
        Some(crate::chat::Usage {
            prompt_tokens: self.0.usage.prompt_tokens as u32,
            completion_tokens: self.0.usage.completion_tokens as u32,
            total_tokens: self.0.usage.total_tokens as u32,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        })
    }
}

#[async_trait]
impl CompletionProvider for MistralRs {
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        if self.is_embedded() {
            return Err(LLMError::ProviderError(
                "Completion endpoint not supported in embedded mode. Use chat instead.".into(),
            ));
        }

        let (base_url, model, api_key) = self
            .get_server_config()
            .ok_or_else(|| LLMError::InvalidRequest("Missing server configuration".to_string()))?;

        let url = format!("{}/v1/completions", base_url);

        let completion_req = serde_json::json!({
            "model": model,
            "prompt": req.prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        });

        let mut request = self.client.post(&url).json(&completion_req);

        if let Some(key) = api_key {
            request = request.bearer_auth(key);
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
        if self.is_embedded() {
            return Err(LLMError::ProviderError(
                "Embeddings not supported in embedded mode. Use server mode with an embedding model.".into(),
            ));
        }

        // Server mode
        let (base_url, model, api_key) = self
            .get_server_config()
            .ok_or_else(|| LLMError::InvalidRequest("Missing server configuration".to_string()))?;

        let url = format!("{}/v1/embeddings", base_url);

        let body = ServerEmbeddingRequest {
            model: model.to_string(),
            input,
        };

        let mut request = self.client.post(&url).json(&body);

        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }

        let resp = request.send().await?.error_for_status()?;
        let json_resp: ServerEmbeddingResponse = resp.json().await?;

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
        if self.is_embedded() {
            // In embedded mode, return the loaded model
            return Ok(Box::new(EmbeddedModelListResponse {
                model_id: match &self.config.mode {
                    MistralRsMode::Embedded { model_id, .. } => model_id.clone(),
                    _ => "unknown".to_string(),
                },
            }));
        }

        // Server mode
        let (base_url, _, api_key) = self
            .get_server_config()
            .ok_or_else(|| LLMError::InvalidRequest("Missing server configuration".to_string()))?;

        let url = format!("{}/v1/models", base_url);

        let mut request = self.client.get(&url);

        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }

        let resp = request.send().await?.error_for_status()?;
        let result: ServerModelListResponse = resp.json().await?;
        Ok(Box::new(result))
    }
}

struct EmbeddedModelListResponse {
    model_id: String,
}

unsafe impl Send for EmbeddedModelListResponse {}
unsafe impl Sync for EmbeddedModelListResponse {}

impl ModelListResponse for EmbeddedModelListResponse {
    fn get_models(&self) -> Vec<String> {
        vec![self.model_id.clone()]
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        vec![Box::new(EmbeddedModelRawEntry {
            id: self.model_id.clone(),
        })]
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::MistralRs
    }
}

#[derive(Debug)]
struct EmbeddedModelRawEntry {
    id: String,
}

impl ModelListRawEntry for EmbeddedModelRawEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        DateTime::<Utc>::UNIX_EPOCH
    }

    fn get_raw(&self) -> Value {
        serde_json::json!({"id": self.id})
    }
}

impl crate::LLMProvider for MistralRs {
    fn tools(&self) -> Option<&[Tool]> {
        self.config.tools.as_deref()
    }
}

/// Parses a Server-Sent Events (SSE) chunk from mistral.rs's streaming API.
fn parse_server_sse(chunk: &str) -> Result<Option<String>, LLMError> {
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

            match serde_json::from_str::<ServerStreamChunk>(data) {
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
