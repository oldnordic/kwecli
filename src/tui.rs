use crossterm::event::KeyCode;
use serde_json;
use futures::executor;

/// Application state and navigation
pub struct App {
    pub current_tab: usize,
    pub tabs: Vec<String>,
    pub input_buffer: String,
    pub chat_history: Vec<(bool, String)>,
    pub analysis_history: Vec<String>,
    pub plan_history: Vec<String>,
    pub config_state: crate::settings::KweConfig,
    client: crate::client::BackendClient,
}

impl App {
    /// Initialize the App with default tabs and client
    pub fn new() -> anyhow::Result<Self> {
        let tabs = vec![
            "Chat".into(),
            "Analysis".into(),
            "Planning".into(),
            "Settings".into(),
        ];
        let config = crate::settings::KweConfig::default();
        let config_client = crate::client::BackendClient::new(&crate::client::Config::default());
        
        // Try to load current settings from backend, but use defaults if it fails
        let cfg = match futures::executor::block_on(config_client.get_config()) {
            Ok(backend_config) => {
                eprintln!("✅ Connected to backend, loaded configuration");
                backend_config
            }
            Err(_) => {
                eprintln!("⚠️  Backend unavailable, using default configuration (offline mode)");
                crate::settings::KweConfig::default()
            }
        };
        
        let client = crate::client::BackendClient::new(&crate::client::Config::default());
        Ok(App {
            current_tab: 0,
            tabs,
            input_buffer: String::new(),
            chat_history: Vec::new(),
            analysis_history: Vec::new(),
            plan_history: Vec::new(),
            config_state: cfg,
            client,
        })
    }

    /// Move to next tab (wrap around)
    pub fn tab_next(&mut self) {
        if !self.tabs.is_empty() {
            self.current_tab = (self.current_tab + 1) % self.tabs.len();
        }
    }

    /// Move to previous tab (wrap around)
    pub fn tab_prev(&mut self) {
        if !self.tabs.is_empty() {
            let len = self.tabs.len();
            self.current_tab = (self.current_tab + len - 1) % len;
        }
    }

    /// Handle key navigation, input editing, and backend calls
    pub fn process_key(&mut self, key: KeyCode) -> anyhow::Result<bool> {
        // Quit on 'q' or Esc
        if matches!(key, KeyCode::Char('q') | KeyCode::Esc) {
            return Ok(false);
        }
        match self.current_tab {
            0 | 1 | 2 => {
                // Chat, Analysis, Planning
                match key {
                    KeyCode::Tab | KeyCode::Right => self.tab_next(),
                    KeyCode::Left => self.tab_prev(),
                    KeyCode::Char(c) => self.input_buffer.push(c),
                    KeyCode::Backspace => { self.input_buffer.pop(); },
                    KeyCode::Enter => {
                        let input = std::mem::take(&mut self.input_buffer);
                        match self.current_tab {
                            0 => {
                                self.chat_history.push((true, input.clone()));
                                let resp = futures::executor::block_on(self.client.send_chat_message(&input))?;
                                self.chat_history.push((false, resp));
                            }
                            1 => {
                                let resp = futures::executor::block_on(self.client.analyze_code(&input))?;
                                self.analysis_history.push(resp);
                            }
                            2 => {
                                let (md, _) = futures::executor::block_on(self.client.plan_generate(&input, None))?;
                                self.plan_history.push(md);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            3 => {
                // Settings: u=toggle Ollama, r=toggle RAG, Enter=set model
                match key {
                    KeyCode::Char('u') => {
                        let new = !self.config_state.use_ollama.unwrap_or(false);
                        let payload = serde_json::json!({"use_ollama": new});
                        let _ = futures::executor::block_on(self.client.update_config(&payload))?;
                        self.config_state.use_ollama = Some(new);
                    }
                    KeyCode::Char('r') => {
                        let new = !self.config_state.rag_enabled.unwrap_or(false);
                        let payload = serde_json::json!({"rag_enabled": new});
                        let _ = futures::executor::block_on(self.client.update_config(&payload))?;
                        self.config_state.rag_enabled = Some(new);
                    }
                    KeyCode::Char(c) => self.input_buffer.push(c),
                    KeyCode::Backspace => { self.input_buffer.pop(); },
                    KeyCode::Enter => {
                        if !self.input_buffer.is_empty() {
                            let model = std::mem::take(&mut self.input_buffer);
                            let payload = serde_json::json!({"ollama_model": model.clone()});
                            let _ = futures::executor::block_on(self.client.update_config(&payload))?;
                            self.config_state.ollama_model = Some(model);
                        }
                    }
                    KeyCode::Tab | KeyCode::Right => self.tab_next(),
                    KeyCode::Left => self.tab_prev(),
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(true)
    }

    /// Run the TUI loop
    pub fn run(&mut self) -> anyhow::Result<()> {
        use crossterm::{execute, terminal::{enable_raw_mode, disable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen}, event::{self, Event}};
        use ratatui::{Terminal, backend::CrosstermBackend, layout::{Layout, Constraint, Direction}, widgets::{Tabs, Block, Borders, List, ListItem, Paragraph}};
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        loop {
            terminal.draw(|f| {
                let size = f.size();
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(3), Constraint::Min(1), Constraint::Length(3)].as_ref())
                    .split(size);
                let tabs = Tabs::new(self.tabs.iter().map(|t| t.as_str()).collect())
                    .block(Block::default().borders(Borders::ALL).title("KWE CLI"))
                    .select(self.current_tab);
                f.render_widget(tabs, chunks[0]);
                match self.current_tab {
                    0 => {
                        let items: Vec<ListItem> = self.chat_history.iter().map(|(u, m)| {
                            let prefix = if *u { "User: " } else { "AI: " };
                            ListItem::new(format!("{}{}", prefix, m))
                        }).collect();
                        let list = List::new(items).block(Block::default().borders(Borders::ALL).title("Chat"));
                        f.render_widget(list, chunks[1]);
                    }
                    1 => {
                        let items: Vec<ListItem> = self.analysis_history.iter().map(|m| ListItem::new(m.clone())).collect();
                        let list = List::new(items).block(Block::default().borders(Borders::ALL).title("Analysis"));
                        f.render_widget(list, chunks[1]);
                    }
                    2 => {
                        let items: Vec<ListItem> = self.plan_history.iter().map(|m| ListItem::new(m.clone())).collect();
                        let list = List::new(items).block(Block::default().borders(Borders::ALL).title("Planning"));
                        f.render_widget(list, chunks[1]);
                    }
                    3 => {
                        let cfg = crate::settings::KweConfig::default();
                        let txt = format!("Ollama: {:?}, RAG: {:?}, Model: {:?}", cfg.use_ollama, cfg.rag_enabled, cfg.ollama_model);
                        let para = Paragraph::new(txt).block(Block::default().borders(Borders::ALL).title("Settings"));
                        f.render_widget(para, chunks[1]);
                    }
                    _ => {}
                }
                let input = Paragraph::new(self.input_buffer.as_str()).block(Block::default().borders(Borders::ALL).title("Input"));
                f.render_widget(input, chunks[2]);
            })?;
            if let Event::Key(key) = event::read()? {
                if !self.process_key(key.code)? { break; }
            }
        }
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }
} // end impl App
