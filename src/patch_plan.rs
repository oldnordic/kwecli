#!/usr/bin/env rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EditItem {
    pub path: String,
    pub new_content: String,
    #[serde(default)]
    pub base_content: Option<String>,
    #[serde(default)]
    pub selected: bool,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PatchPlan {
    pub edits: Vec<EditItem>,
}

impl PatchPlan {
    pub fn load_from_json_str(&mut self, s: &str) -> anyhow::Result<()> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        self.load_from_value(&v)
    }

    pub fn load_from_value(&mut self, v: &serde_json::Value) -> anyhow::Result<()> {
        self.edits.clear();
        if let Some(arr) = v.as_array() {
            for it in arr {
                let path = it.get("path").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("missing path"))?;
                let new_c = it.get("new_content").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("missing new_content"))?;
                let base = it.get("base_content").and_then(|v| v.as_str()).map(|s| s.to_string());
                self.edits.push(EditItem { path: path.to_string(), new_content: new_c.to_string(), base_content: base, selected: true });
            }
        } else {
            // Expect object with key "edits"
            let arr = v.get("edits").and_then(|v| v.as_array()).ok_or_else(|| anyhow::anyhow!("missing edits array"))?;
            for it in arr {
                let path = it.get("path").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("missing path"))?;
                let new_c = it.get("new_content").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("missing new_content"))?;
                let base = it.get("base_content").and_then(|v| v.as_str()).map(|s| s.to_string());
                self.edits.push(EditItem { path: path.to_string(), new_content: new_c.to_string(), base_content: base, selected: true });
            }
        }
        Ok(())
    }

    pub fn toggle(&mut self, idx: usize) -> bool {
        if let Some(e) = self.edits.get_mut(idx) { e.selected = !e.selected; true } else { false }
    }

    pub fn selected_count(&self) -> usize { self.edits.iter().filter(|e| e.selected).count() }

    pub fn to_payload_selected(&self) -> serde_json::Value {
        let mut out: Vec<serde_json::Value> = Vec::new();
        for e in &self.edits {
            if e.selected {
                let mut o = serde_json::Map::new();
                o.insert("path".to_string(), serde_json::json!(e.path));
                o.insert("new_content".to_string(), serde_json::json!(e.new_content));
                if let Some(b) = &e.base_content { o.insert("base_content".to_string(), serde_json::json!(b)); }
                out.push(serde_json::Value::Object(o));
            }
        }
        serde_json::json!({"edits": out})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_and_toggle_and_payload() {
        let json = r#"{"edits":[{"path":"a.py","new_content":"print(1)"},{"path":"b.py","new_content":"print(2)","base_content":"x"}]}"#;
        let mut plan = PatchPlan::default();
        plan.load_from_json_str(json).unwrap();
        assert_eq!(plan.edits.len(), 2);
        assert_eq!(plan.selected_count(), 2);
        assert!(plan.toggle(1));
        assert_eq!(plan.selected_count(), 1);
        let payload = plan.to_payload_selected();
        let arr = payload.get("edits").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].get("path").unwrap().as_str().unwrap(), "a.py");
    }
}

