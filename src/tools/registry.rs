//! Tool registry for managing available tools.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::llm::ToolDefinition;
use crate::tools::builtin::{EchoTool, HttpTool, JsonTool, TimeTool};
use crate::tools::tool::Tool;
use crate::tools::wasm::{
    Capabilities, ResourceLimits, WasmError, WasmStorageError, WasmToolRuntime, WasmToolStore,
    WasmToolWrapper,
};

/// Registry of available tools.
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tool.
    pub async fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.write().await.insert(name.clone(), tool);
        tracing::debug!("Registered tool: {}", name);
    }

    /// Register a tool (sync version for startup).
    pub fn register_sync(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        if let Ok(mut tools) = self.tools.try_write() {
            tools.insert(name.clone(), tool);
            tracing::debug!("Registered tool: {}", name);
        }
    }

    /// Unregister a tool.
    pub async fn unregister(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.write().await.remove(name)
    }

    /// Get a tool by name.
    pub async fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.read().await.get(name).cloned()
    }

    /// Check if a tool exists.
    pub async fn has(&self, name: &str) -> bool {
        self.tools.read().await.contains_key(name)
    }

    /// List all tool names.
    pub async fn list(&self) -> Vec<String> {
        self.tools.read().await.keys().cloned().collect()
    }

    /// Get the number of registered tools.
    pub fn count(&self) -> usize {
        self.tools.try_read().map(|t| t.len()).unwrap_or(0)
    }

    /// Get all tools.
    pub async fn all(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.read().await.values().cloned().collect()
    }

    /// Get tool definitions for LLM function calling.
    pub async fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .read()
            .await
            .values()
            .map(|tool| ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            })
            .collect()
    }

    /// Get tool definitions for specific tools.
    pub async fn tool_definitions_for(&self, names: &[&str]) -> Vec<ToolDefinition> {
        let tools = self.tools.read().await;
        names
            .iter()
            .filter_map(|name| tools.get(*name))
            .map(|tool| ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            })
            .collect()
    }

    /// Register all built-in tools.
    pub fn register_builtin_tools(&self) {
        self.register_sync(Arc::new(EchoTool));
        self.register_sync(Arc::new(TimeTool));
        self.register_sync(Arc::new(JsonTool));
        self.register_sync(Arc::new(HttpTool::new()));

        tracing::info!("Registered {} built-in tools", self.count());
    }

    /// Register a WASM tool from bytes.
    ///
    /// This validates and compiles the WASM component, then registers it as a tool.
    /// The tool will be executed in a sandboxed environment with the given capabilities.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let runtime = Arc::new(WasmToolRuntime::new(WasmRuntimeConfig::default())?);
    /// let wasm_bytes = std::fs::read("my_tool.wasm")?;
    ///
    /// registry.register_wasm(WasmToolRegistration {
    ///     name: "my_tool",
    ///     wasm_bytes: &wasm_bytes,
    ///     runtime: &runtime,
    ///     description: Some("My custom tool description"),
    ///     ..Default::default()
    /// }).await?;
    /// ```
    pub async fn register_wasm(&self, reg: WasmToolRegistration<'_>) -> Result<(), WasmError> {
        // Prepare the module (validates and compiles)
        let prepared = reg
            .runtime
            .prepare(reg.name, reg.wasm_bytes, reg.limits)
            .await?;

        // Create the wrapper
        let mut wrapper = WasmToolWrapper::new(Arc::clone(reg.runtime), prepared, reg.capabilities);

        // Apply overrides if provided
        if let Some(desc) = reg.description {
            wrapper = wrapper.with_description(desc);
        }
        if let Some(s) = reg.schema {
            wrapper = wrapper.with_schema(s);
        }

        // Register the tool
        self.register(Arc::new(wrapper)).await;

        tracing::info!(name = reg.name, "Registered WASM tool");
        Ok(())
    }

    /// Register a WASM tool from database storage.
    ///
    /// Loads the WASM binary with integrity verification and configures capabilities.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = PostgresWasmToolStore::new(pool);
    /// let runtime = Arc::new(WasmToolRuntime::new(WasmRuntimeConfig::default())?);
    ///
    /// registry.register_wasm_from_storage(
    ///     &store,
    ///     &runtime,
    ///     "user_123",
    ///     "my_tool",
    /// ).await?;
    /// ```
    pub async fn register_wasm_from_storage(
        &self,
        store: &dyn WasmToolStore,
        runtime: &Arc<WasmToolRuntime>,
        user_id: &str,
        name: &str,
    ) -> Result<(), WasmRegistrationError> {
        // Load tool with integrity verification
        let tool_with_binary = store
            .get_with_binary(user_id, name)
            .await
            .map_err(WasmRegistrationError::Storage)?;

        // Load capabilities
        let stored_caps = store
            .get_capabilities(tool_with_binary.tool.id)
            .await
            .map_err(WasmRegistrationError::Storage)?;

        let capabilities = stored_caps.map(|c| c.to_capabilities()).unwrap_or_default();

        // Register the tool
        self.register_wasm(WasmToolRegistration {
            name: &tool_with_binary.tool.name,
            wasm_bytes: &tool_with_binary.wasm_binary,
            runtime,
            capabilities,
            limits: None,
            description: Some(&tool_with_binary.tool.description),
            schema: Some(tool_with_binary.tool.parameters_schema.clone()),
        })
        .await
        .map_err(WasmRegistrationError::Wasm)?;

        tracing::info!(
            name = tool_with_binary.tool.name,
            user_id = user_id,
            trust_level = %tool_with_binary.tool.trust_level,
            "Registered WASM tool from storage"
        );

        Ok(())
    }
}

/// Error when registering a WASM tool from storage.
#[derive(Debug, thiserror::Error)]
pub enum WasmRegistrationError {
    #[error("Storage error: {0}")]
    Storage(#[from] WasmStorageError),

    #[error("WASM error: {0}")]
    Wasm(#[from] WasmError),
}

/// Configuration for registering a WASM tool.
pub struct WasmToolRegistration<'a> {
    /// Unique name for the tool.
    pub name: &'a str,
    /// Raw WASM component bytes.
    pub wasm_bytes: &'a [u8],
    /// WASM runtime for compilation and execution.
    pub runtime: &'a Arc<WasmToolRuntime>,
    /// Security capabilities to grant the tool.
    pub capabilities: Capabilities,
    /// Optional resource limits (uses defaults if None).
    pub limits: Option<ResourceLimits>,
    /// Optional description override.
    pub description: Option<&'a str>,
    /// Optional parameter schema override.
    pub schema: Option<serde_json::Value>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tool::EchoTool;

    #[tokio::test]
    async fn test_register_and_get() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool)).await;

        assert!(registry.has("echo").await);
        assert!(registry.get("echo").await.is_some());
        assert!(registry.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_list_tools() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool)).await;

        let tools = registry.list().await;
        assert!(tools.contains(&"echo".to_string()));
    }

    #[tokio::test]
    async fn test_tool_definitions() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool)).await;

        let defs = registry.tool_definitions().await;
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
    }
}
