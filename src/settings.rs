use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct KweConfig {
    pub use_ollama: Option<bool>,
    pub ollama_model: Option<String>,
    pub rag_enabled: Option<bool>,
    pub docs_cache_path: Option<String>,
    pub backend_host: Option<String>,
    pub backend_port: Option<u16>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PendingChanges {
    pub use_ollama: Option<bool>,
    pub rag_enabled: Option<bool>,
    pub ollama_model: Option<String>,
    pub docs_cache_path: Option<String>,
}

impl PendingChanges {
    pub fn new() -> Self { Self::default() }

    pub fn toggle_ollama(&mut self, current: bool) {
        self.use_ollama = Some(!current);
    }

    pub fn toggle_rag(&mut self, current: bool) {
        self.rag_enabled = Some(!current);
    }

    pub fn set_model<S: Into<String>>(&mut self, model: S) {
        self.ollama_model = Some(model.into());
    }

    pub fn set_docs_path<S: Into<String>>(&mut self, path: S) {
        self.docs_cache_path = Some(path.into());
    }

    pub fn to_update_payload(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        if let Some(v) = self.use_ollama { map.insert("use_ollama".to_string(), serde_json::json!(v)); }
        if let Some(v) = self.rag_enabled { map.insert("rag_enabled".to_string(), serde_json::json!(v)); }
        if let Some(ref v) = self.ollama_model { map.insert("ollama_model".to_string(), serde_json::json!(v)); }
        if let Some(ref v) = self.docs_cache_path { map.insert("docs_cache_path".to_string(), serde_json::json!(v)); }
        serde_json::Value::Object(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toggle_ollama_and_rag_and_model() {
        let mut changes = PendingChanges::new();
        changes.toggle_ollama(true); // should set to false
        changes.toggle_rag(false); // should set to true
        changes.set_model("qwen2.5-coder:7b");
        let payload = changes.to_update_payload();
        assert_eq!(payload["use_ollama"].as_bool(), Some(false));
        assert_eq!(payload["rag_enabled"].as_bool(), Some(true));
        assert_eq!(payload["ollama_model"].as_str(), Some("qwen2.5-coder:7b"));
    }

    #[test]
    fn test_docs_path_update_payload_only_changed_keys() {
        let mut changes = PendingChanges::new();
        changes.set_docs_path("./docs_cache");
        let payload = changes.to_update_payload();
        // Only docs_cache_path should be present
        assert!(payload.get("use_ollama").is_none());
        assert!(payload.get("rag_enabled").is_none());
        assert!(payload.get("ollama_model").is_none());
        assert_eq!(payload["docs_cache_path"].as_str(), Some("./docs_cache"));
    }
}

