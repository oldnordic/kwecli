//! In-memory key-value store implementation for SynapseDB

use std::collections::HashMap;

/// In-memory cache implementation
#[derive(Debug, Clone)]
pub struct Cache {
    data: HashMap<String, Vec<u8>>,
}

impl Cache {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Get value by key
    pub fn get(&self, key: &str) -> Option<&Vec<u8>> {
        self.data.get(key)
    }

    /// Set value for key
    pub fn set(&mut self, key: &str, value: Vec<u8>) {
        self.data.insert(key.to_string(), value);
    }

    /// Delete value by key
    pub fn delete(&mut self, key: &str) -> bool {
        self.data.remove(key).is_some()
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cache() {
        let cache = Cache::new();
        assert!(cache.data.is_empty());
    }

    #[test]
    fn test_set_and_get() {
        let mut cache = Cache::new();
        let key = "test_key";
        let value = b"test_value".to_vec();

        cache.set(key, value.clone());
        assert_eq!(cache.get(key), Some(&value));
    }

    #[test]
    fn test_delete() {
        let mut cache = Cache::new();
        let key = "test_key";
        let value = b"test_value".to_vec();

        cache.set(key, value);
        assert!(cache.delete(key));
        assert_eq!(cache.get(key), None);
    }

    #[test]
    fn test_contains_key() {
        let mut cache = Cache::new();
        let key = "test_key";
        let value = b"test_value".to_vec();

        assert!(!cache.contains_key(key));
        cache.set(key, value);
        assert!(cache.contains_key(key));
    }

    #[test]
    fn test_keys() {
        let mut cache = Cache::new();
        cache.set("key1", b"value1".to_vec());
        cache.set("key2", b"value2".to_vec());

        let keys = cache.keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
    }

    #[test]
    fn test_clear() {
        let mut cache = Cache::new();
        cache.set("key1", b"value1".to_vec());
        cache.set("key2", b"value2".to_vec());

        assert_eq!(cache.keys().len(), 2);
        cache.clear();
        assert_eq!(cache.keys().len(), 0);
    }

    #[test]
    fn test_get_nonexistent() {
        let cache = Cache::new();
        assert_eq!(cache.get("nonexistent"), None);
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut cache = Cache::new();
        assert!(!cache.delete("nonexistent"));
    }
}