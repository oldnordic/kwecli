use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::client::BackendClient;

#[derive(Debug, Default, Clone)]
pub struct GitUi {
    pub status_files: Vec<String>,
    pub branches: Vec<String>,
    pub current_branch: String,
    pub diff_text: String,
    pub cwd: Option<String>,
    pub status_message: String,
    pub selected_branch_index: Option<usize>,
}

impl GitUi {
    pub fn new() -> Self {
        Self { status_message: "Git ready".to_string(), ..Default::default() }
    }

    pub async fn refresh_status(&mut self, client: &BackendClient) {
        match client.git_status(self.cwd.as_deref()).await {
            Ok(json) => {
                let mut files = Vec::new();
                if let Some(arr) = json.get("files").and_then(|v| v.as_array()) {
                    for f in arr {
                        let st = f.get("status").and_then(|v| v.as_str()).unwrap_or("");
                        let p = f.get("path").and_then(|v| v.as_str()).unwrap_or("");
                        files.push(format!("{:2} {}", st, p));
                    }
                }
                self.status_files = files;
                self.status_message = "Status refreshed".to_string();
            }
            Err(e) => self.status_message = format!("git status error: {}", e),
        }
    }

    pub async fn refresh_branches(&mut self, client: &BackendClient) {
        match client.git_branches(self.cwd.as_deref()).await {
            Ok(json) => {
                let mut list = Vec::new();
                let mut cur = String::new();
                if let Some(arr) = json.get("branches").and_then(|v| v.as_array()) {
                    for b in arr {
                        let name = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let current = b.get("current").and_then(|v| v.as_bool()).unwrap_or(false);
                        if current { cur = name.to_string(); }
                        list.push(name.to_string());
                    }
                }
                self.branches = list;
                self.current_branch = cur;
                self.status_message = "Branches refreshed".to_string();
                self.selected_branch_index = None;
            }
            Err(e) => self.status_message = format!("git branches error: {}", e),
        }
    }

    pub async fn stage_all(&mut self, client: &BackendClient) {
        match client.git_add(self.cwd.as_deref(), vec![".".to_string()]).await {
            Ok(j) => {
                if j.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
                    self.status_message = "Staged all".to_string();
                } else {
                    self.status_message = format!("Stage failed: {}", j.get("error").and_then(|v| v.as_str()).unwrap_or("unknown"));
                }
            }
            Err(e) => self.status_message = format!("git add error: {}", e),
        }
    }

    pub async fn load_diff(&mut self, client: &BackendClient, staged: bool) {
        match client.git_diff(self.cwd.as_deref(), staged).await {
            Ok(json) => {
                self.diff_text = json.get("diff").and_then(|v| v.as_str()).unwrap_or("").to_string();
                self.status_message = if staged { "Loaded staged diff".to_string() } else { "Loaded diff".to_string() };
            }
            Err(e) => self.status_message = format!("git diff error: {}", e),
        }
    }

    pub async fn commit(&mut self, client: &BackendClient, message: &str) {
        match client.git_commit(self.cwd.as_deref(), message).await {
            Ok(json) => {
                if json.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
                    self.status_message = "Commit OK".to_string();
                } else {
                    self.status_message = format!("Commit failed: {}", json.get("error").and_then(|v| v.as_str()).unwrap_or("unknown"));
                }
            }
            Err(e) => self.status_message = format!("git commit error: {}", e),
        }
    }

    pub async fn checkout(&mut self, client: &BackendClient, name: &str, create: bool) {
        match client.git_checkout(self.cwd.as_deref(), name, create).await {
            Ok(json) => {
                if json.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
                    self.status_message = if create { "Branch created & switched".to_string() } else { "Switched branch".to_string() };
                } else {
                    self.status_message = format!("Checkout failed: {}", json.get("error").and_then(|v| v.as_str()).unwrap_or("unknown"));
                }
            }
            Err(e) => self.status_message = format!("git checkout error: {}", e),
        }
    }

    pub fn branch_name_by_index(&self, idx: usize) -> Option<String> {
        self.branches.get(idx).cloned()
    }

    pub async fn checkout_by_index(&mut self, client: &BackendClient, idx: usize) {
        if let Some(name) = self.branch_name_by_index(idx) {
            self.checkout(client, &name, false).await;
        } else {
            self.status_message = format!("Invalid branch index: {}", idx);
        }
    }

    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
            .split(area);

        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(8), Constraint::Length(8), Constraint::Min(0)])
            .split(cols[0]);

        let status_items: Vec<ListItem> = self
            .status_files
            .iter()
            .map(|s| ListItem::new(s.clone()))
            .collect();
        let status_list = List::new(status_items)
            .block(Block::default().borders(Borders::ALL).title("Status (l=refresh, S=stage all)"))
            .style(Style::default().fg(Color::White));
        frame.render_widget(status_list, left[0]);

        let branches_items: Vec<ListItem> = self
            .branches
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let mut label = format!("[{}] {}", i, b);
                if b == &self.current_branch { label = format!("* {}", label); }
                ListItem::new(label)
            })
            .collect();
        let branches_list = List::new(branches_items)
            .block(Block::default().borders(Borders::ALL).title("Branches (b=refresh, o=checkout, O=create, I=checkout by index)"))
            .style(Style::default().fg(Color::White));
        frame.render_widget(branches_list, left[1]);

        let info = Paragraph::new(self.status_message.clone())
            .block(Block::default().borders(Borders::ALL).title("Git Info (C=commit, d=diff, D=staged diff)"))
            .style(Style::default().fg(Color::White));
        frame.render_widget(info, left[2]);

        let diff = Paragraph::new(self.diff_text.clone())
            .block(Block::default().borders(Borders::ALL).title("Diff"))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });
        frame.render_widget(diff, cols[1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_branch_by_index_and_checkout_invalid() {
        let mut ui = GitUi::new();
        ui.branches = vec!["main".into(), "feature/x".into()];
        assert_eq!(ui.branch_name_by_index(1).as_deref(), Some("feature/x"));
        assert!(ui.branch_name_by_index(5).is_none());
        // checkout_by_index with invalid should not panic and should set status message
        let cfg = crate::client::Config::default();
        let client = crate::client::BackendClient::new(&cfg);
        ui.checkout_by_index(&client, 999).await; // Will just set status message locally
        assert!(ui.status_message.contains("Invalid branch index"));
    }
}
