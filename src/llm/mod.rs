//! LLM integration for the agent.
//!
//! Uses the NEAR AI chat-api as the unified LLM provider.

mod nearai;
mod provider;
mod reasoning;

pub use nearai::NearAiProvider;
pub use provider::{
    ChatMessage, CompletionRequest, CompletionResponse, LlmProvider, Role, ToolCall,
    ToolCompletionRequest, ToolCompletionResponse, ToolDefinition, ToolResult,
};
pub use reasoning::{ActionPlan, Reasoning, ReasoningContext, ToolSelection};

use std::sync::Arc;

use crate::config::LlmConfig;
use crate::error::LlmError;

/// Create an LLM provider based on configuration.
pub fn create_llm_provider(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>, LlmError> {
    Ok(Arc::new(NearAiProvider::new(config.nearai.clone())))
}
