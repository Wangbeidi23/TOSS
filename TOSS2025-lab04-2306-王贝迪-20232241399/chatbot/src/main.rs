use clap::Parser;
mod cli;
mod database;
mod embeddings; // New: Add embeddings module
mod llm; // New: Add llm module

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for better debugging (optional, but good practice)
    tracing_subscriber::fmt::init();

    println!("Hello, Chacha is starting up!");

    let args = cli::Cli::parse();

    match args.command {
        cli::Commands::Ask { query } => {
            println!("Searching for relevant information...");
            let references = database::retrieve(&query).await?;
            if references.is_empty() {
                println!("No relevant information found in the knowledge base. Answering based on general knowledge...");
            } else {
                println!("Found relevant information:");
                for ref_content in &references {
                    println!("  - {}", ref_content.content);
                }
            }
            let answer = llm::answer_with_context(&query, references).await?;
            println!("Answer: {}", answer);
        },
        cli::Commands::Remember { content } => {
            println!("Remembering: \"{}\"...", content);
            let inserted_content = database::insert(&content).await?;
            println!("Successfully remembered content with ID: {}", inserted_content.id);
        }
    }
    Ok(())
}
