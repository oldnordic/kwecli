#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanningState {
    pub goal: String,
    pub constraints: Option<String>,
    pub markdown: Option<String>,
    pub parsed_present: bool,
    pub last_error: Option<String>,
}

impl PlanningState {
    pub fn set_goal<S: Into<String>>(&mut self, goal: S) { self.goal = goal.into(); }
    pub fn set_constraints<S: Into<String>>(&mut self, c: S) { self.constraints = Some(c.into()); }
    pub fn clear_constraints(&mut self) { self.constraints = None; }
    pub fn clear(&mut self) {
        self.markdown = None;
        self.parsed_present = false;
        self.last_error = None;
    }
    pub fn apply_generated(&mut self, markdown: String, parsed_present: bool) {
        self.markdown = Some(markdown);
        self.parsed_present = parsed_present;
        self.last_error = None;
    }
    pub fn set_error<S: Into<String>>(&mut self, err: S) { self.last_error = Some(err.into()); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_and_constraints_flow() {
        let mut st = PlanningState::default();
        st.set_goal("Build feature X");
        st.set_constraints("No network; Python only");
        assert_eq!(st.goal, "Build feature X");
        assert_eq!(st.constraints.as_deref(), Some("No network; Python only"));

        st.apply_generated("## Plan\n1) Do A".to_string(), true);
        assert!(st.markdown.as_ref().unwrap().starts_with("## Plan"));
        assert!(st.parsed_present);

        st.clear();
        assert!(st.markdown.is_none());
        assert!(!st.parsed_present);
    }
}

