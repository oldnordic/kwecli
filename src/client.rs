use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use chrono::Utc;

/// Configuration for BackendClient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub python_backend_url: String,
}

impl Default for Config {
    fn default() -> Self {
        let host = std::env::var("KWE_BACKEND_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = std::env::var("KWE_BACKEND_PORT").unwrap_or_else(|_| "8000".to_string());
        Config { python_backend_url: format!("http://{}:{}", host, port) }
    }
}

/// HTTP client for backend communication
#[derive(Debug, Clone)]
pub struct BackendClient {
    http_client: Client,
    base_url: String,
}

impl BackendClient {
    /// Create a new BackendClient with timeout and resilience
    pub fn new(config: &Config) -> Self {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(2))  // 2 second timeout
            .connect_timeout(std::time::Duration::from_secs(1))  // 1 second connect timeout
            .build()
            .expect("Failed to build HTTP client");
        BackendClient { http_client, base_url: config.python_backend_url.clone() }
    }

    /// Send a chat message to the backend and return the response text
    pub async fn send_chat_message(&self, message: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);
        let body = serde_json::json!({
            "prompt": message,
            "context": "cli_interaction",
            "timestamp": Utc::now().to_rfc3339(),
        });
        let resp = self.http_client.post(&url)
            .json(&body)
            .send().await
            .context("POST /api/chat failed")?;
        let json: serde_json::Value = resp.json().await.context("Parsing JSON response failed")?;
        Ok(json.get("response").and_then(|v| v.as_str()).unwrap_or_default().to_string())
    }
    /// Return the full chat endpoint URL
    pub fn chat_url(&self) -> String {
        format!("{}/api/chat", self.base_url)
    }
    /// Analyze code for patterns and improvements
    pub async fn analyze_code(&self, file_path: &str) -> Result<String> {
        let url = format!("{}/api/analyze", self.base_url);
        let body = serde_json::json!({
            "file": file_path,
            "timestamp": Utc::now().to_rfc3339(),
        });
        let resp = self.http_client.post(&url)
            .json(&body)
            .send().await
            .context("POST /api/analyze failed")?;
        let json: serde_json::Value = resp.json().await.context("Parsing analyze JSON response failed")?;
        Ok(json.get("analysis").and_then(|v| v.as_str()).unwrap_or_default().to_string())
    }

    /// Generate a project plan based on description and constraints
    pub async fn plan_generate(&self, goal: &str, constraints: Option<&str>) -> Result<(String, serde_json::Value)> {
        let url = format!("{}/api/plan/generate", self.base_url);
        let body = serde_json::json!({
            "goal": goal,
            "constraints": constraints.unwrap_or(""),
        });
        let resp = self.http_client.post(&url)
            .json(&body)
            .send().await
            .context("POST /api/plan/generate failed")?;
        let parsed: serde_json::Value = resp.json().await.context("Parsing plan JSON response failed")?;
        let markdown = parsed.get("markdown").and_then(|v| v.as_str()).unwrap_or_default().to_string();
        Ok((markdown, parsed))
    }
    /// Update backend settings
    pub async fn update_config(&self, payload: &serde_json::Value) -> Result<serde_json::Value> {
        let url = format!("{}/api/config/update", self.base_url);
        let resp = self.http_client.post(&url)
            .json(payload)
            .send().await
            .context("POST /api/config/update failed")?;
        let json = resp.json().await.context("Parsing update JSON response failed")?;
        Ok(json)
    }
    /// Get current config from backend with graceful fallback
    pub async fn get_config(&self) -> Result<crate::settings::KweConfig> {
        let url = format!("{}/api/config/get", self.base_url);
        
        // Try to get config from backend, but fallback to default if it fails
        match self.http_client.get(&url).send().await {
            Ok(resp) => {
                match resp.json::<crate::settings::KweConfig>().await {
                    Ok(config) => Ok(config),
                    Err(_) => {
                        // Backend responded but with invalid JSON, use default
                        eprintln!("Warning: Backend returned invalid config, using defaults");
                        Ok(crate::settings::KweConfig::default())
                    }
                }
            }
            Err(_) => {
                // Backend is unreachable, use default config for offline mode
                eprintln!("Warning: Backend unreachable at {}, running in offline mode", self.base_url);
                Ok(crate::settings::KweConfig::default())
            }
        }
    }
    
    /// Check if backend is available
    pub async fn is_backend_available(&self) -> bool {
        let url = format!("{}/api/config/get", self.base_url);
        match self.http_client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_url_constructed() {
        let config = Config { python_backend_url: "http://example.com".into() };
        let client = BackendClient::new(&config);
        assert_eq!(client.chat_url(), "http://example.com/api/chat");
    }
}
