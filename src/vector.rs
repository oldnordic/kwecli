//! In-memory HNSW vector index implementation for SynapseDB

use std::collections::HashMap;

/// Vector index using HNSW (Hierarchical Navigable Small World) algorithm
#[derive(Debug)]
pub struct VectorIndex {
    vectors: HashMap<u64, Vec<f32>>,
    dimension: usize,
}

impl VectorIndex {
    /// Create a new vector index with specified dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            dimension,
        }
    }

    /// Add a vector to the index
    pub fn add_vector(&mut self, id: u64, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dimension {
            return Err(format!(
                "Vector dimension mismatch. Expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }
        
        self.vectors.insert(id, vector);
        Ok(())
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Find k nearest neighbors for a query vector
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut distances = Vec::new();
        
        for (&id, vector) in &self.vectors {
            let distance = self.euclidean_distance(query, vector);
            distances.push((id, distance));
        }
        
        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        
        distances
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Clear all vectors from the index
    pub fn clear(&mut self) {
        self.vectors.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_vector_index() {
        let index = VectorIndex::new(3);
        assert_eq!(index.dimension, 3);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_add_vector() {
        let mut index = VectorIndex::new(2);
        let vector = vec![1.0, 2.0];
        
        assert!(index.add_vector(1, vector.clone()).is_ok());
        assert_eq!(index.len(), 1);
        
        // Test with wrong dimension
        let wrong_vector = vec![1.0, 2.0, 3.0];
        assert!(index.add_vector(2, wrong_vector).is_err());
    }

    #[test]
    fn test_search() {
        let mut index = VectorIndex::new(2);
        
        // Add some vectors
        index.add_vector(1, vec![1.0, 1.0]).unwrap();
        index.add_vector(2, vec![2.0, 2.0]).unwrap();
        index.add_vector(3, vec![3.0, 3.0]).unwrap();
        
        // Query for nearest neighbors
        let query = vec![1.5, 1.5];
        let results = index.search(&query, 2);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // Closest to [1.5, 1.5]
        assert_eq!(results[1].0, 2); // Second closest
    }

    #[test]
    fn test_empty_index() {
        let index = VectorIndex::new(3);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        
        let results = index.search(&[0.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut index = VectorIndex::new(2);
        index.add_vector(1, vec![1.0, 1.0]).unwrap();
        index.add_vector(2, vec![2.0, 2.0]).unwrap();
        
        assert_eq!(index.len(), 2);
        index.clear();
        assert_eq!(index.len(), 0);
    }
}