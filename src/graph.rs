//! In-memory graph representation for SynapseDB

use std::collections::{HashMap, HashSet};

/// Node in the graph with associated data
#[derive(Debug, Clone)]
pub struct Node {
    pub id: u64,
    pub data: serde_json::Value,
}

/// Graph implementation using adjacency list representation
#[derive(Debug)]
pub struct Graph {
    nodes: HashMap<u64, Node>,
    edges: HashMap<u64, HashSet<u64>>,
}

impl Graph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, id: u64, data: serde_json::Value) -> Result<(), String> {
        if self.nodes.contains_key(&id) {
            return Err(format!("Node with ID {} already exists", id));
        }
        
        self.nodes.insert(id, Node { id, data });
        Ok(())
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: u64, to: u64) -> Result<(), String> {
        // Check if both nodes exist
        if !self.nodes.contains_key(&from) {
            return Err(format!("Node with ID {} does not exist", from));
        }
        
        if !self.nodes.contains_key(&to) {
            return Err(format!("Node with ID {} does not exist", to));
        }
        
        // Add edge
        self.edges.entry(from).or_insert_with(HashSet::new).insert(to);
        Ok(())
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node_id: u64) -> Option<&HashSet<u64>> {
        self.edges.get(&node_id)
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Get a specific node by ID
    pub fn get_node(&self, id: u64) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(HashSet::len).sum()
    }

    /// Clear all nodes and edges from the graph
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_new_graph() {
        let graph = Graph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut graph = Graph::new();
        let data = json!({"name": "Alice"});
        
        assert!(graph.add_node(1, data.clone()).is_ok());
        assert_eq!(graph.node_count(), 1);
        
        // Try to add duplicate node
        assert!(graph.add_node(1, data).is_err());
    }

    #[test]
    fn test_add_edge() {
        let mut graph = Graph::new();
        let data = json!({"name": "Alice"});
        
        // Add nodes first
        graph.add_node(1, data.clone()).unwrap();
        graph.add_node(2, data.clone()).unwrap();
        
        // Add edge
        assert!(graph.add_edge(1, 2).is_ok());
        assert_eq!(graph.edge_count(), 1);
        
        // Try to add edge from non-existent node
        assert!(graph.add_edge(3, 2).is_err());
    }

    #[test]
    fn test_neighbors() {
        let mut graph = Graph::new();
        let data = json!({"name": "Alice"});
        
        graph.add_node(1, data.clone()).unwrap();
        graph.add_node(2, data.clone()).unwrap();
        graph.add_node(3, data).unwrap();
        
        graph.add_edge(1, 2).unwrap();
        graph.add_edge(1, 3).unwrap();
        
        let neighbors = graph.neighbors(1).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
    }

    #[test]
    fn test_clear() {
        let mut graph = Graph::new();
        let data = json!({"name": "Alice"});
        
        graph.add_node(1, data.clone()).unwrap();
        graph.add_node(2, data).unwrap();
        graph.add_edge(1, 2).unwrap();
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        
        graph.clear();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }
}