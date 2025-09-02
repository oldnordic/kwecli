use anyhow::Result;
use clap::{Arg, Command};
use kwecli::tui::App;

#[tokio::main]
async fn main() -> Result<()> {
    let matches = Command::new("kwecli")
        .version("0.1.0")
        .about("KWE CLI - Autonomous development tool with LTMC integration")
        .arg(
            Arg::new("help")
                .short('h')
                .long("help")
                .action(clap::ArgAction::SetTrue)
                .help("Print help information")
        )
        .arg(
            Arg::new("version")
                .short('V')
                .long("version")
                .action(clap::ArgAction::SetTrue)
                .help("Print version information")
        )
        .arg(
            Arg::new("offline")
                .long("offline")
                .action(clap::ArgAction::SetTrue)
                .help("Run in offline mode without backend")
        )
        .get_matches();

    // Handle version flag
    if matches.get_flag("version") {
        println!("kwecli {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    // Handle help flag (clap handles this automatically, but just in case)
    if matches.get_flag("help") {
        return Ok(());
    }

    // Initialize and run the TUI frontend
    println!("🚀 Starting KWE CLI...");
    
    if matches.get_flag("offline") {
        println!("📴 Running in offline mode");
    }

    let mut app = App::new()?;
    app.run()?;
    Ok(())
}
