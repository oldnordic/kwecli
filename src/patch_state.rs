#[derive(Debug, Clone, Default, PartialEq)]
pub struct PatchState {
    pub target_file: String,
    pub new_content_path: String,
    pub last_diff: String,
}

impl PatchState {
    pub fn set_target<S: Into<String>>(&mut self, path: S) { self.target_file = path.into(); }
    pub fn set_content_path<S: Into<String>>(&mut self, path: S) { self.new_content_path = path.into(); }
    pub fn set_diff<S: Into<String>>(&mut self, diff: S) { self.last_diff = diff.into(); }
    pub fn clear(&mut self) { self.target_file.clear(); self.new_content_path.clear(); self.last_diff.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_state_flow() {
        let mut st = PatchState::default();
        st.set_target("src/lib.rs");
        st.set_content_path("/tmp/new.txt");
        st.set_diff("--- a\n+++ b\n");
        assert_eq!(st.target_file, "src/lib.rs");
        assert_eq!(st.new_content_path, "/tmp/new.txt");
        assert!(st.last_diff.starts_with("--- a"));
        st.clear();
        assert_eq!(st.target_file, "");
        assert_eq!(st.last_diff, "");
    }
}

