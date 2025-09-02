#[derive(Debug, Clone, PartialEq)]
pub struct GuidedState {
    pub description: String,
    pub language: String,
    pub code_path: String,
    pub test_path: String,
    pub last_result: String,
}

impl Default for GuidedState {
    fn default() -> Self {
        Self {
            description: String::new(),
            language: "python".to_string(),
            code_path: "generated_code.py".to_string(),
            test_path: "tests/test_generated_code.py".to_string(),
            last_result: String::new(),
        }
    }
}

impl GuidedState {
    pub fn set_description<S: Into<String>>(&mut self, d: S) { self.description = d.into(); }
    pub fn set_language<S: Into<String>>(&mut self, l: S) { self.language = l.into(); }
    pub fn set_code_path<S: Into<String>>(&mut self, p: S) { self.code_path = p.into(); }
    pub fn set_test_path<S: Into<String>>(&mut self, p: S) { self.test_path = p.into(); }
    pub fn clear_result(&mut self) { self.last_result.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guided_state_defaults_and_updates() {
        let mut st = GuidedState::default();
        assert_eq!(st.language, "python");
        st.set_description("Build CLI parser");
        st.set_language("rust");
        st.set_code_path("src/new.rs");
        st.set_test_path("tests/test_new.rs");
        assert_eq!(st.description, "Build CLI parser");
        assert_eq!(st.language, "rust");
        assert_eq!(st.code_path, "src/new.rs");
        assert_eq!(st.test_path, "tests/test_new.rs");
    }
}

