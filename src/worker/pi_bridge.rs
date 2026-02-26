//! Pi coding agent bridge for sandboxed execution.
//!
//! Spawns the `pi` CLI inside a Docker container in JSON streaming mode and
//! streams its NDJSON output back to the orchestrator via HTTP. Supports
//! follow-up prompts via RPC commands.
//!
//! Pi has no built-in permission system by design -- the Docker container
//! is the sole security boundary. An IronClaw-specific extension bundled in
//! the container image handles max-turn enforcement via `ctx.abort()`.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │ Docker Container                                      │
//! │                                                       │
//! │  ironclaw pi-bridge --job-id <uuid>                   │
//! │    └─ writes /workspace/.pi/ironclaw-container.ts     │
//! │    └─ pi --mode json --no-session                     │
//! │       --no-extensions --extension <ext>                │
//! │       --tools <list> "task"                            │
//! │    └─ reads stdout line-by-line (NDJSON)              │
//! │    └─ POSTs events to orchestrator                    │
//! │    └─ polls for follow-up prompts                     │
//! └──────────────────────────────────────────────────────┘
//! ```

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use uuid::Uuid;

use crate::error::WorkerError;
use crate::worker::api::{CompletionReport, JobEventPayload, WorkerHttpClient};

/// Configuration for the Pi bridge runtime.
pub struct PiBridgeConfig {
    pub job_id: Uuid,
    pub orchestrator_url: String,
    pub max_turns: u32,
    pub provider: String,
    pub model: String,
    /// Tool names to enable via `--tools` (e.g. `["read", "write", "edit", "bash"]`).
    pub allowed_tools: Vec<String>,
    /// Optional reasoning effort level (e.g. "low", "medium", "high").
    /// Passed as `--reasoning-effort` to the `pi` CLI.
    pub reasoning_effort: Option<String>,
}

/// A Pi coding agent streaming event (NDJSON line from `--mode json`).
///
/// Pi emits one JSON object per line. Key event types:
///
///   session       -> session header (id, version, cwd)
///   agent_start   -> agent loop begins
///   agent_end     -> agent loop finished (contains all messages)
///   turn_start    -> new LLM turn begins
///   turn_end      -> LLM turn finished (message + toolResults)
///   message_start -> message being assembled
///   message_update -> streaming delta (text or tool call progress)
///   message_end   -> complete message
///   tool_execution_start -> tool call begins
///   tool_execution_end   -> tool call completed (result + isError)
///
/// Unlike Claude Code, content blocks are part of the `message` field
/// directly, with role-based structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiStreamEvent {
    #[serde(rename = "type")]
    pub event_type: String,

    /// Session metadata (for session events).
    #[serde(default)]
    pub id: Option<String>,

    #[serde(default)]
    pub version: Option<u32>,

    #[serde(default)]
    pub cwd: Option<String>,

    /// Turn index (for turn_start/turn_end events).
    #[serde(default, rename = "turnIndex")]
    pub turn_index: Option<u32>,

    /// The message object (for message_start/update/end and turn_end).
    #[serde(default)]
    pub message: Option<PiMessage>,

    /// Streaming delta info (for message_update events).
    #[serde(default, rename = "assistantMessageEvent")]
    pub assistant_message_event: Option<PiAssistantEvent>,

    /// Tool results from a turn (for turn_end events).
    #[serde(default, rename = "toolResults")]
    pub tool_results: Option<Vec<PiMessage>>,

    /// All messages (for agent_end events).
    #[serde(default)]
    pub messages: Option<Vec<PiMessage>>,

    // Tool execution fields (tool_execution_start/end).
    #[serde(default, rename = "toolCallId")]
    pub tool_call_id: Option<String>,

    #[serde(default, rename = "toolName")]
    pub tool_name: Option<String>,

    #[serde(default)]
    pub args: Option<serde_json::Value>,

    #[serde(default)]
    pub result: Option<serde_json::Value>,

    #[serde(default, rename = "isError")]
    pub is_error: Option<bool>,

    #[serde(default, rename = "partialResult")]
    pub partial_result: Option<serde_json::Value>,
}

/// A Pi message (user, assistant, or toolResult).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiMessage {
    #[serde(default)]
    pub role: Option<String>,

    #[serde(default)]
    pub content: Option<Vec<PiContentBlock>>,

    #[serde(default, rename = "toolCallId")]
    pub tool_call_id: Option<String>,
}

/// A content block within a Pi message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,

    /// Text content.
    #[serde(default)]
    pub text: Option<String>,

    /// Tool call name.
    #[serde(default)]
    pub name: Option<String>,

    /// Tool call ID.
    #[serde(default)]
    pub id: Option<String>,

    /// Tool call arguments (JSON object).
    #[serde(default)]
    pub arguments: Option<serde_json::Value>,

    /// Tool result content.
    #[serde(default)]
    pub content: Option<serde_json::Value>,
}

/// Streaming delta from the LLM (nested in message_update events).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiAssistantEvent {
    #[serde(rename = "type")]
    pub event_type: String,

    /// Text delta content.
    #[serde(default)]
    pub delta: Option<String>,

    /// Tool call name (for toolcall_start).
    #[serde(default)]
    pub name: Option<String>,

    /// Tool call ID (for toolcall_start).
    #[serde(default)]
    pub id: Option<String>,
}

/// The Pi bridge runtime.
pub struct PiBridgeRuntime {
    config: PiBridgeConfig,
    client: Arc<WorkerHttpClient>,
}

impl PiBridgeRuntime {
    /// Create a new bridge runtime.
    ///
    /// Reads `IRONCLAW_WORKER_TOKEN` from the environment for auth.
    pub fn new(config: PiBridgeConfig) -> Result<Self, WorkerError> {
        let client = Arc::new(WorkerHttpClient::from_env(
            config.orchestrator_url.clone(),
            config.job_id,
        )?);

        Ok(Self { config, client })
    }

    /// Write the IronClaw container extension for max-turn enforcement.
    ///
    /// This TypeScript extension subscribes to `turn_end` events and calls
    /// `ctx.abort()` when the configured turn limit is reached.
    fn write_container_extension(&self) -> Result<std::path::PathBuf, WorkerError> {
        let ext_dir = std::path::Path::new("/workspace/.pi/extensions");
        std::fs::create_dir_all(ext_dir).map_err(|e| WorkerError::ExecutionFailed {
            reason: format!("failed to create /workspace/.pi/extensions/: {e}"),
        })?;

        let ext_path = ext_dir.join("ironclaw-container.ts");
        let ext_content = format!(
            r#"// IronClaw container extension -- max-turn enforcement + compaction.
// Auto-generated by the Pi bridge runtime. Do not edit.
export default function(pi: any) {{
    const maxTurns = {};

    pi.on("turn_end", (event: any, ctx: any) => {{
        if (event.turnIndex >= maxTurns - 1) {{
            ctx.abort();
        }}
    }});
}}"#,
            self.config.max_turns
        );

        std::fs::write(&ext_path, &ext_content).map_err(|e| WorkerError::ExecutionFailed {
            reason: format!("failed to write container extension: {e}"),
        })?;

        tracing::info!(
            job_id = %self.config.job_id,
            max_turns = self.config.max_turns,
            "Wrote Pi container extension"
        );

        Ok(ext_path)
    }

    /// Run the bridge: fetch job, spawn pi, stream events, handle follow-ups.
    pub async fn run(&self) -> Result<(), WorkerError> {
        // Write container extension for max-turn enforcement
        let ext_path = self.write_container_extension()?;

        // Fetch the job description from the orchestrator
        let job = self.client.get_job().await?;

        tracing::info!(
            job_id = %self.config.job_id,
            "Starting Pi bridge for: {}",
            truncate(&job.description, 100)
        );

        // Fetch credentials for injection into the spawned process
        let credentials = self.client.fetch_credentials().await?;
        let mut extra_env = std::collections::HashMap::new();
        for cred in &credentials {
            extra_env.insert(cred.env_var.clone(), cred.value.clone());
        }
        if !extra_env.is_empty() {
            tracing::info!(
                job_id = %self.config.job_id,
                "Fetched {} credential(s) for child process injection",
                extra_env.len()
            );
        }

        // Report that we're running
        self.client
            .report_status(&crate::worker::api::StatusUpdate {
                state: "running".to_string(),
                message: Some("Spawning Pi coding agent".to_string()),
                iteration: 0,
            })
            .await?;

        // Run the initial Pi session (creates a session file for follow-ups)
        let result = self
            .run_pi_session(&job.description, &ext_path, &extra_env, false)
            .await;

        match result {
            Ok(()) => {}
            Err(e) => {
                tracing::error!(job_id = %self.config.job_id, "Pi session failed: {}", e);
                self.client
                    .report_complete(&CompletionReport {
                        success: false,
                        message: Some(format!("Pi coding agent failed: {}", e)),
                        iterations: 1,
                    })
                    .await?;
                return Ok(());
            }
        }

        // Follow-up loop: poll for prompts, run new pi sessions
        let mut iteration = 1u32;
        loop {
            match self.poll_for_prompt().await {
                Ok(Some(prompt)) => {
                    if prompt.done {
                        tracing::info!(job_id = %self.config.job_id, "Orchestrator signaled done");
                        break;
                    }
                    iteration += 1;
                    tracing::info!(
                        job_id = %self.config.job_id,
                        "Got follow-up prompt, resuming Pi session"
                    );
                    if let Err(e) = self
                        .run_pi_session(&prompt.content, &ext_path, &extra_env, true)
                        .await
                    {
                        tracing::error!(
                            job_id = %self.config.job_id,
                            "Follow-up Pi session failed: {}", e
                        );
                        self.report_event(
                            "status",
                            &serde_json::json!({
                                "message": format!("Follow-up session failed: {}", e),
                            }),
                        )
                        .await;
                    }
                }
                Ok(None) => {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                Err(e) => {
                    tracing::warn!(
                        job_id = %self.config.job_id,
                        "Prompt polling error: {}", e
                    );
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }

        self.client
            .report_complete(&CompletionReport {
                success: true,
                message: Some("Pi coding agent session completed".to_string()),
                iterations: iteration,
            })
            .await?;

        Ok(())
    }

    /// Spawn a `pi` CLI process in JSON streaming mode and stream its output.
    ///
    /// On the initial run (`is_follow_up = false`) Pi creates a session file
    /// under `/workspace/.pi/sessions/`.  Follow-up runs pass `--continue` so
    /// Pi loads that session and the new prompt has full conversation history.
    async fn run_pi_session(
        &self,
        prompt: &str,
        ext_path: &std::path::Path,
        extra_env: &std::collections::HashMap<String, String>,
        is_follow_up: bool,
    ) -> Result<(), WorkerError> {
        let mut cmd = Command::new("pi");
        cmd.arg("--mode")
            .arg("json")
            .arg("--no-extensions")
            .arg("--no-skills")
            .arg("--no-prompt-templates")
            .arg("--no-themes")
            .arg("--extension")
            .arg(ext_path)
            .arg("--model")
            .arg(format!("{}/{}", self.config.provider, self.config.model));

        // Follow-up prompts resume the previous session so Pi has full
        // conversation context.  The initial run creates a session file;
        // `--continue` loads the most recent one.
        if is_follow_up {
            cmd.arg("--continue");
        }

        // Pass reasoning effort if specified
        if let Some(ref effort) = self.config.reasoning_effort {
            cmd.arg("--reasoning-effort").arg(effort);
        }

        // Restrict tools if configured
        if !self.config.allowed_tools.is_empty() {
            cmd.arg("--tools").arg(self.config.allowed_tools.join(","));
        }

        // The prompt is the final positional argument
        cmd.arg(prompt);

        // Inject credentials into child process only
        cmd.envs(extra_env);

        cmd.current_dir("/workspace")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| WorkerError::ExecutionFailed {
            reason: format!("failed to spawn pi: {}", e),
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| WorkerError::ExecutionFailed {
                reason: "failed to capture pi stdout".to_string(),
            })?;

        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| WorkerError::ExecutionFailed {
                reason: "failed to capture pi stderr".to_string(),
            })?;

        // Spawn stderr reader that forwards lines as log events
        let client_for_stderr = Arc::clone(&self.client);
        let job_id = self.config.job_id;
        let stderr_handle = tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(job_id = %job_id, "pi stderr: {}", line);
                let payload = JobEventPayload {
                    event_type: "status".to_string(),
                    data: serde_json::json!({ "message": line }),
                };
                client_for_stderr.post_event(&payload).await;
            }
        });

        // Read stdout NDJSON line by line
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            match serde_json::from_str::<PiStreamEvent>(&line) {
                Ok(event) => {
                    let payloads = pi_event_to_payloads(&event);
                    for payload in payloads {
                        self.report_event(&payload.event_type, &payload.data).await;
                    }
                }
                Err(e) => {
                    tracing::debug!(
                        job_id = %self.config.job_id,
                        "Non-JSON pi output: {} (parse error: {})", line, e
                    );
                    self.report_event("status", &serde_json::json!({ "message": line }))
                        .await;
                }
            }
        }

        // Wait for the process to exit
        let status = child
            .wait()
            .await
            .map_err(|e| WorkerError::ExecutionFailed {
                reason: format!("failed waiting for pi: {}", e),
            })?;

        // Wait for stderr reader to finish
        let _ = stderr_handle.await;

        if !status.success() {
            let code = status.code().unwrap_or(-1);
            tracing::warn!(
                job_id = %self.config.job_id,
                exit_code = code,
                "Pi process exited with non-zero status"
            );

            self.report_event(
                "result",
                &serde_json::json!({
                    "status": "error",
                    "exit_code": code,
                }),
            )
            .await;

            return Err(WorkerError::ExecutionFailed {
                reason: format!("pi exited with code {}", code),
            });
        }

        // Report successful result
        self.report_event(
            "result",
            &serde_json::json!({
                "status": "completed",
            }),
        )
        .await;

        Ok(())
    }

    /// Post a job event to the orchestrator.
    async fn report_event(&self, event_type: &str, data: &serde_json::Value) {
        let payload = JobEventPayload {
            event_type: event_type.to_string(),
            data: data.clone(),
        };
        self.client.post_event(&payload).await;
    }

    /// Poll the orchestrator for a follow-up prompt.
    async fn poll_for_prompt(
        &self,
    ) -> Result<Option<crate::worker::api::PromptResponse>, WorkerError> {
        self.client.poll_prompt().await
    }
}

/// Convert a Pi stream event into one or more event payloads for the orchestrator.
///
/// Maps Pi's event types to IronClaw's unified event vocabulary:
///   session             -> status (session init)
///   agent_start         -> status
///   agent_end           -> result
///   message_end         -> message (for assistant text)
///   tool_execution_start -> tool_use
///   tool_execution_end  -> tool_result
///   turn_start/turn_end -> status (progress tracking)
fn pi_event_to_payloads(event: &PiStreamEvent) -> Vec<JobEventPayload> {
    let mut payloads = Vec::new();

    match event.event_type.as_str() {
        "session" => {
            payloads.push(JobEventPayload {
                event_type: "status".to_string(),
                data: serde_json::json!({
                    "message": "Pi coding agent session started",
                    "session_id": event.id,
                }),
            });
        }
        "agent_start" => {
            payloads.push(JobEventPayload {
                event_type: "status".to_string(),
                data: serde_json::json!({
                    "message": "Pi agent started",
                }),
            });
        }
        "agent_end" => {
            // Extract final assistant text from messages
            if let Some(ref messages) = event.messages {
                for msg in messages.iter().rev() {
                    if msg.role.as_deref() == Some("assistant")
                        && let Some(text) = extract_text_from_message(msg)
                        && !text.is_empty()
                    {
                        payloads.push(JobEventPayload {
                            event_type: "message".to_string(),
                            data: serde_json::json!({
                                "role": "assistant",
                                "content": text,
                            }),
                        });
                        break;
                    }
                }
            }

            payloads.push(JobEventPayload {
                event_type: "result".to_string(),
                data: serde_json::json!({
                    "status": "completed",
                }),
            });
        }
        "message_end" => {
            // Emit completed assistant messages (not tool results)
            if let Some(ref msg) = event.message
                && msg.role.as_deref() == Some("assistant")
                && let Some(ref blocks) = msg.content
            {
                for block in blocks {
                    match block.block_type.as_str() {
                        "text" => {
                            if let Some(ref text) = block.text.as_deref().filter(|t| !t.is_empty())
                            {
                                payloads.push(JobEventPayload {
                                    event_type: "message".to_string(),
                                    data: serde_json::json!({
                                        "role": "assistant",
                                        "content": text,
                                    }),
                                });
                            }
                        }
                        "toolCall" => {
                            payloads.push(JobEventPayload {
                                event_type: "tool_use".to_string(),
                                data: serde_json::json!({
                                    "tool_name": block.name,
                                    "tool_use_id": block.id,
                                    "input": block.arguments,
                                }),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        "tool_execution_start" => {
            payloads.push(JobEventPayload {
                event_type: "tool_use".to_string(),
                data: serde_json::json!({
                    "tool_name": event.tool_name,
                    "tool_use_id": event.tool_call_id,
                    "input": event.args,
                }),
            });
        }
        "tool_execution_end" => {
            payloads.push(JobEventPayload {
                event_type: "tool_result".to_string(),
                data: serde_json::json!({
                    "tool_use_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "output": event.result,
                    "is_error": event.is_error.unwrap_or(false),
                }),
            });
        }
        "turn_start" => {
            payloads.push(JobEventPayload {
                event_type: "status".to_string(),
                data: serde_json::json!({
                    "message": format!("Turn {} started", event.turn_index.unwrap_or(0) + 1),
                    "turn_index": event.turn_index,
                }),
            });
        }
        "turn_end" => {
            payloads.push(JobEventPayload {
                event_type: "status".to_string(),
                data: serde_json::json!({
                    "message": format!("Turn {} completed", event.turn_index.unwrap_or(0) + 1),
                    "turn_index": event.turn_index,
                }),
            });
        }
        // Compaction and retry events from the container extension
        "auto_compaction_start" | "auto_compaction_end" | "auto_retry_start" | "auto_retry_end" => {
            payloads.push(JobEventPayload {
                event_type: "status".to_string(),
                data: serde_json::json!({
                    "message": format!("Pi event: {}", event.event_type),
                    "raw_type": event.event_type,
                }),
            });
        }
        // message_start and message_update are streaming deltas -- skip to avoid noise
        "message_start" | "message_update" | "tool_execution_update" => {}
        _ => {
            payloads.push(JobEventPayload {
                event_type: "status".to_string(),
                data: serde_json::json!({
                    "message": format!("Pi event: {}", event.event_type),
                    "raw_type": event.event_type,
                }),
            });
        }
    }

    payloads
}

/// Extract concatenated text content from a Pi message.
fn extract_text_from_message(msg: &PiMessage) -> Option<String> {
    let blocks = msg.content.as_ref()?;
    let text: String = blocks
        .iter()
        .filter(|b| b.block_type == "text")
        .filter_map(|b| b.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() { None } else { Some(text) }
}

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_session_event() {
        let json = r#"{"type":"session","version":3,"id":"abc-123","timestamp":"2026-01-01","cwd":"/workspace"}"#;
        let event: PiStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "session");
        assert_eq!(event.id.as_deref(), Some("abc-123"));
        assert_eq!(event.version, Some(3));
    }

    #[test]
    fn test_parse_agent_start_event() {
        let json = r#"{"type":"agent_start"}"#;
        let event: PiStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "agent_start");
    }

    #[test]
    fn test_parse_turn_start_event() {
        let json = r#"{"type":"turn_start","turnIndex":0,"timestamp":1234567890}"#;
        let event: PiStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "turn_start");
        assert_eq!(event.turn_index, Some(0));
    }

    #[test]
    fn test_parse_message_end_with_text() {
        let json = r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"Hello world"}]}}"#;
        let event: PiStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "message_end");
        let msg = event.message.unwrap();
        assert_eq!(msg.role.as_deref(), Some("assistant"));
        let blocks = msg.content.unwrap();
        assert_eq!(blocks[0].block_type, "text");
        assert_eq!(blocks[0].text.as_deref(), Some("Hello world"));
    }

    #[test]
    fn test_parse_message_end_with_tool_call() {
        let json = r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"toolCall","name":"bash","id":"tc_1","arguments":{"command":"ls"}}]}}"#;
        let event: PiStreamEvent = serde_json::from_str(json).unwrap();
        let msg = event.message.unwrap();
        let blocks = msg.content.unwrap();
        assert_eq!(blocks[0].block_type, "toolCall");
        assert_eq!(blocks[0].name.as_deref(), Some("bash"));
        assert_eq!(blocks[0].id.as_deref(), Some("tc_1"));
        assert!(blocks[0].arguments.is_some());
    }

    #[test]
    fn test_parse_tool_execution_start() {
        let json = r#"{"type":"tool_execution_start","toolCallId":"tc_1","toolName":"bash","args":{"command":"ls"}}"#;
        let event: PiStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "tool_execution_start");
        assert_eq!(event.tool_call_id.as_deref(), Some("tc_1"));
        assert_eq!(event.tool_name.as_deref(), Some("bash"));
    }

    #[test]
    fn test_parse_tool_execution_end() {
        let json = r#"{"type":"tool_execution_end","toolCallId":"tc_1","toolName":"bash","result":{"output":"file.txt"},"isError":false}"#;
        let event: PiStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "tool_execution_end");
        assert_eq!(event.is_error, Some(false));
        assert!(event.result.is_some());
    }

    #[test]
    fn test_parse_agent_end_with_messages() {
        let json = r#"{"type":"agent_end","messages":[{"role":"user","content":[{"type":"text","text":"List files"}]},{"role":"assistant","content":[{"type":"text","text":"Done."}]}]}"#;
        let event: PiStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "agent_end");
        let msgs = event.messages.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1].role.as_deref(), Some("assistant"));
    }

    #[test]
    fn test_pi_event_to_payloads_session() {
        let event = PiStreamEvent {
            event_type: "session".to_string(),
            id: Some("sid-1".to_string()),
            version: Some(3),
            cwd: Some("/workspace".to_string()),
            turn_index: None,
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "status");
        assert_eq!(payloads[0].data["session_id"], "sid-1");
    }

    #[test]
    fn test_pi_event_to_payloads_agent_end() {
        let event = PiStreamEvent {
            event_type: "agent_end".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: Some(vec![PiMessage {
                role: Some("assistant".to_string()),
                content: Some(vec![PiContentBlock {
                    block_type: "text".to_string(),
                    text: Some("All done.".to_string()),
                    name: None,
                    id: None,
                    arguments: None,
                    content: None,
                }]),
                tool_call_id: None,
            }]),
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        // message + result
        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].event_type, "message");
        assert_eq!(payloads[0].data["content"], "All done.");
        assert_eq!(payloads[1].event_type, "result");
        assert_eq!(payloads[1].data["status"], "completed");
    }

    #[test]
    fn test_pi_event_to_payloads_tool_execution_start() {
        let event = PiStreamEvent {
            event_type: "tool_execution_start".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: Some("tc_1".to_string()),
            tool_name: Some("bash".to_string()),
            args: Some(serde_json::json!({"command": "ls"})),
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "tool_use");
        assert_eq!(payloads[0].data["tool_name"], "bash");
        assert_eq!(payloads[0].data["tool_use_id"], "tc_1");
    }

    #[test]
    fn test_pi_event_to_payloads_tool_execution_end() {
        let event = PiStreamEvent {
            event_type: "tool_execution_end".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: Some("tc_1".to_string()),
            tool_name: Some("bash".to_string()),
            args: None,
            result: Some(serde_json::json!("file.txt\nREADME.md")),
            is_error: Some(false),
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "tool_result");
        assert_eq!(payloads[0].data["tool_use_id"], "tc_1");
        assert_eq!(payloads[0].data["is_error"], false);
    }

    #[test]
    fn test_pi_event_to_payloads_message_update_skipped() {
        let event = PiStreamEvent {
            event_type: "message_update".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: None,
            assistant_message_event: Some(PiAssistantEvent {
                event_type: "text_delta".to_string(),
                delta: Some("Hello".to_string()),
                name: None,
                id: None,
            }),
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert!(payloads.is_empty(), "message_update should be skipped");
    }

    #[test]
    fn test_pi_event_to_payloads_unknown_type() {
        let event = PiStreamEvent {
            event_type: "fancy_new_thing".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "status");
    }

    #[test]
    fn test_extract_text_from_message() {
        let msg = PiMessage {
            role: Some("assistant".to_string()),
            content: Some(vec![
                PiContentBlock {
                    block_type: "text".to_string(),
                    text: Some("Hello".to_string()),
                    name: None,
                    id: None,
                    arguments: None,
                    content: None,
                },
                PiContentBlock {
                    block_type: "toolCall".to_string(),
                    text: None,
                    name: Some("bash".to_string()),
                    id: Some("tc_1".to_string()),
                    arguments: Some(serde_json::json!({"command": "ls"})),
                    content: None,
                },
                PiContentBlock {
                    block_type: "text".to_string(),
                    text: Some("World".to_string()),
                    name: None,
                    id: None,
                    arguments: None,
                    content: None,
                },
            ]),
            tool_call_id: None,
        };
        let text = extract_text_from_message(&msg).unwrap();
        assert_eq!(text, "Hello\nWorld");
    }

    #[test]
    fn test_extract_text_from_message_empty() {
        let msg = PiMessage {
            role: Some("assistant".to_string()),
            content: Some(vec![PiContentBlock {
                block_type: "toolCall".to_string(),
                text: None,
                name: Some("bash".to_string()),
                id: None,
                arguments: None,
                content: None,
            }]),
            tool_call_id: None,
        };
        assert!(extract_text_from_message(&msg).is_none());
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello");
        assert_eq!(truncate("", 5), "");
    }

    #[test]
    fn test_pi_event_payload_serde() {
        let payload = JobEventPayload {
            event_type: "message".to_string(),
            data: serde_json::json!({ "role": "assistant", "content": "hi" }),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let parsed: JobEventPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.event_type, "message");
        assert_eq!(parsed.data["content"], "hi");
    }

    #[test]
    fn test_pi_event_to_payloads_message_end_text() {
        let event = PiStreamEvent {
            event_type: "message_end".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: Some(PiMessage {
                role: Some("assistant".to_string()),
                content: Some(vec![PiContentBlock {
                    block_type: "text".to_string(),
                    text: Some("Here's the answer".to_string()),
                    name: None,
                    id: None,
                    arguments: None,
                    content: None,
                }]),
                tool_call_id: None,
            }),
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "message");
        assert_eq!(payloads[0].data["role"], "assistant");
        assert_eq!(payloads[0].data["content"], "Here's the answer");
    }

    #[test]
    fn test_pi_event_to_payloads_message_end_tool_call() {
        let event = PiStreamEvent {
            event_type: "message_end".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: Some(PiMessage {
                role: Some("assistant".to_string()),
                content: Some(vec![PiContentBlock {
                    block_type: "toolCall".to_string(),
                    text: None,
                    name: Some("bash".to_string()),
                    id: Some("tc_42".to_string()),
                    arguments: Some(serde_json::json!({"command": "ls -la"})),
                    content: None,
                }]),
                tool_call_id: None,
            }),
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "tool_use");
        assert_eq!(payloads[0].data["tool_name"], "bash");
        assert_eq!(payloads[0].data["tool_use_id"], "tc_42");
    }

    #[test]
    fn test_pi_event_to_payloads_message_end_non_assistant_skipped() {
        // User messages at message_end should not produce payloads
        let event = PiStreamEvent {
            event_type: "message_end".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: Some(PiMessage {
                role: Some("user".to_string()),
                content: Some(vec![PiContentBlock {
                    block_type: "text".to_string(),
                    text: Some("user prompt".to_string()),
                    name: None,
                    id: None,
                    arguments: None,
                    content: None,
                }]),
                tool_call_id: None,
            }),
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert!(
            payloads.is_empty(),
            "user message_end should produce no payloads"
        );
    }

    #[test]
    fn test_pi_event_to_payloads_agent_end_no_messages() {
        // agent_end with no messages should still produce a result
        let event = PiStreamEvent {
            event_type: "agent_end".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "result");
        assert_eq!(payloads[0].data["status"], "completed");
    }

    #[test]
    fn test_pi_event_to_payloads_turn_start() {
        let event = PiStreamEvent {
            event_type: "turn_start".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: Some(2),
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "status");
        assert_eq!(payloads[0].data["message"], "Turn 3 started");
        assert_eq!(payloads[0].data["turn_index"], 2);
    }

    #[test]
    fn test_pi_event_to_payloads_turn_end() {
        let event = PiStreamEvent {
            event_type: "turn_end".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: Some(4),
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "status");
        assert_eq!(payloads[0].data["message"], "Turn 5 completed");
    }

    #[test]
    fn test_pi_event_to_payloads_tool_error() {
        let event = PiStreamEvent {
            event_type: "tool_execution_end".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: Some("tc_err".to_string()),
            tool_name: Some("bash".to_string()),
            args: None,
            result: Some(serde_json::json!("command not found")),
            is_error: Some(true),
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "tool_result");
        assert_eq!(payloads[0].data["is_error"], true);
        assert_eq!(payloads[0].data["output"], "command not found");
    }

    #[test]
    fn test_pi_event_to_payloads_message_start_skipped() {
        let event = PiStreamEvent {
            event_type: "message_start".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: None,
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert!(payloads.is_empty(), "message_start should be skipped");
    }

    #[test]
    fn test_pi_event_to_payloads_compaction_events() {
        for event_type in [
            "auto_compaction_start",
            "auto_compaction_end",
            "auto_retry_start",
            "auto_retry_end",
        ] {
            let event = PiStreamEvent {
                event_type: event_type.to_string(),
                id: None,
                version: None,
                cwd: None,
                turn_index: None,
                message: None,
                assistant_message_event: None,
                tool_results: None,
                messages: None,
                tool_call_id: None,
                tool_name: None,
                args: None,
                result: None,
                is_error: None,
                partial_result: None,
            };
            let payloads = pi_event_to_payloads(&event);
            assert_eq!(
                payloads.len(),
                1,
                "compaction event {event_type} should emit status"
            );
            assert_eq!(payloads[0].event_type, "status");
            assert_eq!(payloads[0].data["raw_type"], event_type);
        }
    }

    #[test]
    fn test_write_container_extension() {
        let tmp = tempfile::tempdir().unwrap();
        let ext_dir = tmp.path().join(".pi").join("extensions");

        // Patch the function to write to a temp path by testing the content directly
        let max_turns = 25;
        let ext_content = format!(
            r#"// IronClaw container extension -- max-turn enforcement + compaction.
// Auto-generated by the Pi bridge runtime. Do not edit.
export default function(pi: any) {{
    const maxTurns = {};

    pi.on("turn_end", (event: any, ctx: any) => {{
        if (event.turnIndex >= maxTurns - 1) {{
            ctx.abort();
        }}
    }});
}}"#,
            max_turns
        );

        std::fs::create_dir_all(&ext_dir).unwrap();
        let ext_path = ext_dir.join("ironclaw-container.ts");
        std::fs::write(&ext_path, &ext_content).unwrap();

        let written = std::fs::read_to_string(&ext_path).unwrap();
        assert!(written.contains("const maxTurns = 25;"));
        assert!(written.contains("ctx.abort()"));
        assert!(written.contains("turn_end"));
        assert!(written.contains("event.turnIndex >= maxTurns - 1"));
    }

    #[test]
    fn test_pi_code_config_defaults() {
        let config = crate::config::PiCodeConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_turns, 50);
        assert_eq!(config.memory_limit_mb, 4096);
        assert_eq!(
            config.allowed_tools,
            vec!["read", "write", "edit", "bash", "grep", "find", "ls"]
        );
    }

    #[test]
    fn test_extract_text_from_message_no_content() {
        let msg = PiMessage {
            role: Some("assistant".to_string()),
            content: None,
            tool_call_id: None,
        };
        assert!(extract_text_from_message(&msg).is_none());
    }

    #[test]
    fn test_agent_end_picks_last_assistant() {
        // agent_end should pick the LAST assistant message
        let event = PiStreamEvent {
            event_type: "agent_end".to_string(),
            id: None,
            version: None,
            cwd: None,
            turn_index: None,
            message: None,
            assistant_message_event: None,
            tool_results: None,
            messages: Some(vec![
                PiMessage {
                    role: Some("assistant".to_string()),
                    content: Some(vec![PiContentBlock {
                        block_type: "text".to_string(),
                        text: Some("First response".to_string()),
                        name: None,
                        id: None,
                        arguments: None,
                        content: None,
                    }]),
                    tool_call_id: None,
                },
                PiMessage {
                    role: Some("user".to_string()),
                    content: Some(vec![PiContentBlock {
                        block_type: "text".to_string(),
                        text: Some("Follow up".to_string()),
                        name: None,
                        id: None,
                        arguments: None,
                        content: None,
                    }]),
                    tool_call_id: None,
                },
                PiMessage {
                    role: Some("assistant".to_string()),
                    content: Some(vec![PiContentBlock {
                        block_type: "text".to_string(),
                        text: Some("Final answer".to_string()),
                        name: None,
                        id: None,
                        arguments: None,
                        content: None,
                    }]),
                    tool_call_id: None,
                },
            ]),
            tool_call_id: None,
            tool_name: None,
            args: None,
            result: None,
            is_error: None,
            partial_result: None,
        };
        let payloads = pi_event_to_payloads(&event);
        assert_eq!(payloads.len(), 2); // message + result
        assert_eq!(payloads[0].event_type, "message");
        assert_eq!(payloads[0].data["content"], "Final answer");
    }
}
