#[derive(Debug, Clone)]
pub struct Selector {
    items: Vec<String>,
    index: usize,
}

impl Selector {
    pub fn new(items: Vec<String>) -> Self { Self { items, index: 0 } }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn index(&self) -> usize { self.index }
    pub fn items(&self) -> &Vec<String> { &self.items }
    pub fn selected(&self) -> Option<&str> { self.items.get(self.index).map(|s| s.as_str()) }
    pub fn next(&mut self) { if !self.items.is_empty() { self.index = (self.index + 1) % self.items.len(); } }
    pub fn prev(&mut self) { if !self.items.is_empty() { self.index = (self.index + self.items.len() - 1) % self.items.len(); } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_navigation() {
        let mut s = Selector::new(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(s.selected(), Some("a"));
        s.next(); assert_eq!(s.selected(), Some("b"));
        s.next(); assert_eq!(s.selected(), Some("c"));
        s.next(); assert_eq!(s.selected(), Some("a"));
        s.prev(); assert_eq!(s.selected(), Some("c"));
    }
}

