//! On-disk temporal logging implementation for SynapseDB

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};

/// Temporal logger that writes events to disk
#[derive(Debug)]
pub struct TemporalLogger {
    file_path: String,
}

impl TemporalLogger {
    /// Create a new temporal logger with given file path
    pub fn new(file_path: &str) -> Self {
        Self {
            file_path: file_path.to_string(),
        }
    }

    /// Log an event to the temporal log
    pub fn log(&mut self, event: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)?;
        
        writeln!(file, "{}", event)?;
        Ok(())
    }

    /// Read all events from the temporal log
    pub fn read_events(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);
        
        let mut events = Vec::new();
        for line in reader.lines() {
            events.push(line?);
        }
        
        Ok(events)
    }

    /// Get the number of events in the log
    pub fn event_count(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);
        
        let mut count = 0;
        for line in reader.lines() {
            line?;
            count += 1;
        }
        
        Ok(count)
    }

    /// Clear the temporal log
    pub fn clear(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.file_path)?;
        
        drop(file); // Close file to ensure truncation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_new_temporal_logger() {
        let logger = TemporalLogger::new("temporal.log");
        assert_eq!(logger.file_path, "temporal.log");
    }

    #[test]
    fn test_log_and_read_events() -> Result<(), Box<dyn std::error::Error>> {
        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path().to_str().unwrap();
        
        let mut logger = TemporalLogger::new(file_path);
        
        // Log some events
        logger.log("Event 1")?;
        logger.log("Event 2")?;
        logger.log("Event 3")?;
        
        // Read them back
        let events = logger.read_events()?;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], "Event 1");
        assert_eq!(events[1], "Event 2");
        assert_eq!(events[2], "Event 3");
        
        Ok(())
    }

    #[test]
    fn test_event_count() -> Result<(), Box<dyn std::error::Error>> {
        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path().to_str().unwrap();
        
        let mut logger = TemporalLogger::new(file_path);
        
        // Log some events
        logger.log("Event 1")?;
        logger.log("Event 2")?;
        
        // Check count
        assert_eq!(logger.event_count()?, 2);
        
        Ok(())
    }

    #[test]
    fn test_clear() -> Result<(), Box<dyn std::error::Error>> {
        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path().to_str().unwrap();
        
        let mut logger = TemporalLogger::new(file_path);
        
        // Log some events
        logger.log("Event 1")?;
        logger.log("Event 2")?;
        
        // Verify they exist
        assert_eq!(logger.event_count()?, 2);
        
        // Clear them
        logger.clear()?;
        
        // Verify they're gone
        assert_eq!(logger.event_count()?, 0);
        
        Ok(())
    }

    #[test]
    fn test_empty_log() -> Result<(), Box<dyn std::error::Error>> {
        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path().to_str().unwrap();
        
        let logger = TemporalLogger::new(file_path);
        
        // Read from empty log
        let events = logger.read_events()?;
        assert_eq!(events.len(), 0);
        
        Ok(())
    }
}