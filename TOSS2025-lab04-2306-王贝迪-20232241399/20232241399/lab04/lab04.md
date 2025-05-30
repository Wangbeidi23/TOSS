---

# 实验四：Rust 聊天机器人 (Chatbot) 实现报告

## 1. 个人信息

*   **班级**: 软2306
*   **姓名**: 王贝迪
*   **学号**: 20232241399

## 2. 实验简介

本次实验旨在从零开始构建一个基于 Rust 语言的聊天机器人 (Chatbot)。该机器人将具备以下核心功能：
1.  **命令行交互**: 通过 `clap` 库处理命令行参数，支持 `ask` (提问) 和 `remember` (记忆) 命令。
2.  **本地知识库**: 利用 `SurrealDB` 存储用户输入的“记忆”内容，并结合嵌入 (embeddings) 技术实现内容检索，构建一个检索增强生成 (RAG) 的知识库。
3.  **大语言模型 (LLM) 集成**: 使用 `candle` 框架加载预训练的 LLM (Phi-2 模型)，并结合检索到的本地知识库内容生成问题的答案。

通过本次实验，我们不仅将学习 Rust 异步编程、依赖管理、命令行解析，还将深入理解 RAG 架构中嵌入、向量检索和 LLM 推理的流程。

## 3. 实验环境与工具

*   **操作系统**: Linux (或 WSL)
*   **编程语言**: Rust (需安装 Rustup)
*   **依赖管理**: Cargo
*   **主要库**:
    *   `clap`: 命令行参数解析
    *   `anyhow`: 错误处理
    *   `tokio`: 异步运行时
    *   `candle-core`, `candle-transformers`, `candle-nn`: 大模型推理框架
    *   `hf-hub`: Hugging Face 模型下载
    *   `tokenizers`: 文本分词
    *   `surrealdb`: 本地嵌入数据库
    *   `lazy_static`, `async_once`: 异步初始化单例
    *   `serde_json`, `serde`: 序列化与反序列化
*   **系统依赖**: `clang` (用于 `surrealdb` 的 RocksDB 后端)

## 4. 实验步骤与实现

### 4.1 项目初始化

1.  **创建新的 Rust 项目**:
    ```bash
    cargo new chatbot
    cd chatbot
    ```

### 4.2 添加项目依赖

编辑 `Cargo.toml` 文件，在 `[dependencies]` 部分添加所有必要的依赖项。

```toml
[Dependencies]
anyhow = "1.0.75"
candle-core = { git = "https://github.com/huggingface/candle", branch = "main" }
candle-transformers = { git = "https://github.com/huggingface/candle", branch = "main" }
candle-nn = { git = "https://github.com/huggingface/candle", branch = "main" }
tokenizers = "0.15.0"
tokio = { version = "1", features = ["full", "macros", "rt-multi-thread"] }
tracing = "0.1"
tracing-subscriber = "0.3"
hf-hub = { version = "0.3.2", features = ["tokio"] }
clap = { version = "4.4.11", features = ["derive"] }
serde_json = "1.0.108"
tracing-chrome = "0.7.1"
lazy_static = "1.4.0"
surrealdb = { version = "1.0.0", features = ["kv-rocksdb"] }
serde = { version = "1.0.193", features = ["derive"] }
async_once = "0.2.6"
reqwest = "0.11.22"
regex = "1.10.2"
chrono = "0.4.31"
pdf-extract = "0.7.2"
byteorder = "1.5.0"
wav = "1.0.0"
rand = "0.8.5"
tempfile = "3.8.0"
dirs = "5.0.1"
prettytable-rs = "0.10.0"
```

### 4.3 修改 `main` 函数为异步

为了支持异步操作（如数据库连接、模型加载和推理），需要将 `main` 函数修改为异步，并使用 `tokio` 运行时宏。

编辑 `src/main.rs`:

```rust
#[tokio::main]
async fn main() {
    println!("Hello, world!");
}
```
此时，可以尝试执行 `cargo run` 验证基本设置。

### 4.4 增加命令行处理

为了使聊天机器人能够通过命令行接收指令，我们引入 `clap` 库来解析命令行参数。

1.  **安装 `clap` 和 `anyhow`**:
    ```bash
    cargo add clap -F derive
    cargo add anyhow
    ```
2.  **创建 `src/cli.rs` 文件**: 定义命令行结构和子命令。
    ```rust
    use clap::{Parser, Subcommand};
    
    #[derive(Debug, Parser)]
    #[command(name = "Chacha")]
    #[command(about = "Chacha is AI assistant which is tailored just for you", long_about = None)]
    pub struct Cli {
        #[command(subcommand)]
        pub command: Commands,
    }
    
    #[derive(Debug, Subcommand)]
    pub enum Commands {
        /// Ask a question
        Ask {
            /// The question to ask
            query: String,
        },
        /// Tell Chacha something to remember
        Remember {
            /// The content to remember
            content: String,
        },
    }
    ```
3.  **修改 `src/main.rs`**: 集成 `cli` 模块，并根据解析的命令执行不同的逻辑。
    ```rust
    use clap::Parser;
    mod cli;
    
    #[tokio::main]
    async fn main() -> anyhow::Result<()> {
        println!("Hello, world!");
        let args = cli::Cli::parse();
        match args.command {
            cli::Commands::Ask { query } => {
                let answer = "haha"; // Placeholder
                println!("Answer: {}", answer);
            },
            cli::Commands::Remember { content } => {
                println!("hey, please remember: {}", content); // Add content to output
            }
        }
        Ok(())
    }
    ```
4.  **测试命令行处理**:
    ```bash
    cargo run -- ask "hihi"
    cargo run -- remember "Rust is fun"
    ```
    预期输出将是 `Answer: haha` 或 `hey, please remember: Rust is fun`。

### 4.5 建立本地知识库 (RAG 核心)

本地知识库的建立是实现检索增强生成的关键部分，它包括两个主要模块：`database.rs` 用于数据存储和检索，`embeddings.rs` 用于生成文本嵌入。

1.  **安装 `libclang`**: 这是 `surrealdb` 使用 `RocksDb` 后端所需的系统依赖。
    ```bash
    sudo apt update && sudo apt install -y clang
    ```
2.  **创建 `src/database.rs`**: 该文件负责与 `SurrealDB` 交互，存储和检索内容及其向量嵌入。
    ```rust
    use anyhow::{Context, Error, Result};
    use serde::{Deserialize, Serialize};
    use surrealdb::engine::local::{Db, RocksDb};
    use surrealdb::sql::{thing, Datetime, Thing, Uuid};
    use surrealdb::Surreal;
    lazy_static::lazy_static! {
        pub static ref DB: async_once::AsyncOnce<Surreal<Db>> =
            async_once::AsyncOnce::new(async {
                let db = connect_db().await.expect("Unable to connect to database");
                db
            });
    }
    
    async fn connect_db() -> Result<Surreal<Db>, Box<dyn std::error::Error>> {
        let db_path = std::env::current_dir().unwrap().join("db");
        let db = Surreal::new::<RocksDb>(db_path).await?;
        db.use_ns("rag").use_db("content").await?;
        Ok(db)
    }
    
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct Content {
        pub id: Thing,
        pub content: String,
        pub vector: Vec<f32>,
        pub created_at: Datetime,
    }
    
    pub async fn retrieve(query: &str) -> Result<Vec<Content>, Error> {
        let embeddings: Vec<f32> =
            crate::embeddings::get_embeddings(&query)?.reshape((384,))?.to_vec1()?;
        let db = DB.get().await.clone();
        let mut result = db
            .query("SELECT *, vector::similarity::cosine(vector, $query) AS score FROM vector_index ORDER BY score DESC LIMIT 4")
            .bind(("query", embeddings))
            .await?;
        let vector_indexes: Vec<Content> = result.take(0)?;
        Ok(vector_indexes)
    }
    
    pub async fn insert(content: &str) -> Result<Content, Error> {
        let db = DB.get().await.clone();
        let id = Uuid::new_v4().0.to_string().replace("-", "");
        let id = thing(format!("vector_index:{}", id).as_str())?;
        let vector =
            crate::embeddings::get_embeddings(&content)?.reshape((384,))?.to_vec1()?;
        let vector_index: Content = db
            .create(("vector_index", id.clone()))
            .content(Content {
                id: id.clone(),
                content: content.to_string(),
                vector,
                created_at: Datetime::default(),
            })
            .await?
            .context("Unable to insert vector index")?;
        Ok(vector_index)
    }
    ```
    *   **`DB`**: 使用 `lazy_static` 和 `async_once` 确保数据库连接只初始化一次。
    *   **`connect_db`**: 连接到 `SurrealDB`，并指定命名空间 `rag` 和数据库 `content`。数据文件将存储在项目根目录下的 `db` 文件夹中。
    *   **`Content` struct**: 定义了存储在数据库中的内容结构，包括 ID、原始文本、嵌入向量和创建时间。
    *   **`retrieve`**: 接收查询字符串，通过 `embeddings` 模块获取其向量，然后在 `vector_index` 表中执行余弦相似度搜索，返回最相关的 4 条内容。
    *   **`insert`**: 接收文本内容，生成其向量嵌入，并将其作为新的 `vector_index` 记录插入数据库。

3.  **创建 `src/embeddings.rs`**: 该文件负责加载预训练的嵌入模型 (BAAI/bge-small-en-v1.5) 并生成文本嵌入。
    ```rust
    use anyhow::{Context, Error as E, Result};
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config, DTYPE};
    use hf_hub::{api::sync::Api, Repo};
    use lazy_static::lazy_static;
    use tokenizers::{PaddingParams, Tokenizer};
    
    lazy_static! {
        pub static ref AI: (BertModel, Tokenizer) =
            load_model().expect("Unable to load model");
    }
    
    pub fn load_model() -> Result<(BertModel, Tokenizer)> {
        let api = Api::new()?.repo(Repo::model("BAAI/bge-small-en-v1.5".to_string()));
        // Fetching the config, tokenizer and weights files
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = api.get("pytorch_model.bin")?;
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb = VarBuilder::from_pth(&weights_filename, DTYPE, &Device::Cpu)?;
        let model = BertModel::load(vb, &config)?;
        // Setting the padding strategy for the tokenizer
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        Ok((model, tokenizer))
    }
    
    pub fn get_embeddings(sentence: &str) -> Result<Tensor> {
        let (model, tokenizer) = &*AI;
        // Tokenizing the sentence
        let tokens = tokenizer.encode_batch(vec![sentence], true).map_err(E::msg).context("Unable to encode sentence")?;
        // Getting the token ids from the tokens
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &Device::Cpu)?)
            })
            .collect::<Result<Vec<_>>>().context("Unable to get token ids")?;
        // Stacking the token ids into a tensor
        let token_ids = Tensor::stack(&token_ids, 0).context("Unable to stack token ids")?;
        let token_type_ids = token_ids.zeros_like().context("Unable to get token type ids")?;
        // Getting the embeddings from the model
        let embeddings = model.forward(&token_ids, &token_type_ids).context("Unable to get embeddings")?;
        // Normalizing the embeddings
        let (_n_sentence, n_tokens, _hidden_size) =
            embeddings.dims3().context("Unable to get embeddings dimensions")?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64)).context("Unable to get embeddings sum")?;
        let embeddings =
            embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?).context("Unable to get embeddings broadcast div")?;
        Ok(embeddings)
    }
    ```
    *   **`AI`**: 使用 `lazy_static` 异步加载嵌入模型和分词器。
    *   **`load_model`**: 从 Hugging Face Hub 下载 `BAAI/bge-small-en-v1.5` 模型的配置文件、分词器和权重，然后加载 `BertModel` 和 `Tokenizer`。
    *   **`get_embeddings`**: 接收句子，通过分词器将其编码为 token ID，然后输入到 `BertModel` 中获取嵌入向量，最后进行归一化处理。

### 4.6 与 LLM 模型连接

LLM 连接模块负责加载大语言模型，并结合检索到的知识库内容生成最终答案。

1.  **创建 `src/llm.rs`**: 该文件将加载 Phi-2 模型，并实现文本生成逻辑。
    ```rust
    // Adopted from https://github.com/huggingface/candle/blob/96f1a28e390fceeaa12b3272c8ac5dccc8eb5fa/candle-examples/examples/phi/main.rs
    use anyhow::{Error as E, Result};
    use candle_core::{DType, Device, Tensor};
    use candle_transformers::generation::LogitsProcessor;
    use candle_transformers::models::quantized_mixformer::Config;
    use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
    use hf_hub::{api::sync::Api, Repo};
    use lazy_static::lazy_static;
    use serde_json::json;
    use tokenizers::Tokenizer;
    use crate::database::Content;
    
    lazy_static! {
        pub static ref PHI: (QMixFormer, Tokenizer) =
            load_model().expect("Unable to load model");
    }
    
    pub fn load_model() -> Result<(QMixFormer, Tokenizer)> {
        let api = Api::new()?.repo(Repo::model("Demonthos/dolphin-2_6-phi-2-candle".to_string()));
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = api.get("model-q4k.gguf")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let config = Config::v2();
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&weights_filename)?;
        let model = QMixFormer::new_v2(&config, vb)?;
        Ok((model, tokenizer))
    }
    
    struct TextGeneration {
        model: QMixFormer,
        device: Device,
        tokenizer: Tokenizer,
        logits_processor: LogitsProcessor,
        repeat_penalty: f32,
        repeat_last_n: usize,
    }
    
    impl TextGeneration {
        #[allow(clippy::too_many_arguments)]
        fn new(
            model: QMixFormer,
            tokenizer: Tokenizer,
            seed: u64,
            temp: Option<f64>,
            top_p: Option<f64>,
            repeat_penalty: f32,
            repeat_last_n: usize,
            device: &Device,
        ) -> Self {
            let logits_processor = LogitsProcessor::new(seed, temp, top_p);
            Self {
                model,
                tokenizer,
                logits_processor,
                repeat_penalty,
                repeat_last_n,
                device: device.clone(),
            }
        }
    
        fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
            // Encode the prompt into tokens
            let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
            let mut tokens = tokens.get_ids().to_vec();
            let eos_token = match self.tokenizer.get_vocab(true).get(" <|im_end|>") {
                Some(token) => *token,
                None => anyhow::bail!("cannot find the endoftext token"),
            };
    
            // Loop over the sample length to generate the response
            let mut response = String::new();
            for index in 0..sample_len {
                // Get the context for the current iteration
                let context_size = if index > 0 { 1 } else { tokens.len() };
                let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
    
                // Run the model forward pass
                let logits = self.model.forward(&input)?;
                let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
    
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                // Apply the repetition penalty
                let logits = candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?;
    
                // Sample the next token
                let next_token = self.logits_processor.sample(&logits)?;
                tokens.push(next_token);
    
                // Check if the generated token is the endoftext token
                if next_token == eos_token {
                    break;
                }
                let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
                response += &token;
            }
            Ok(response.trim().to_string())
        }
    }
    
    pub async fn answer_with_context(query: &str, references: Vec<Content>) -> Result<String> {
        // Create the context for the prompt
        let mut context = Vec::new();
        for reference in references.clone() {
            context.push(json!({"content": reference.content}))
        }
        let context = json!(context).to_string();
    
        // Create the prompt
        let prompt = format!(
            "<|im_start|>system\nAs a friendly and helpful AI assistant named Chacha. Your answer should be very concise and to the point. Do not repeat question or references. <|im_end|>\n<|im_start|>user\nquestion: \"{question}\"\nreferences: \"{context}\"\n<|im_end|>\n<|im_start|>assistant\n",
            context = context,
            question = query
        );
        let (model, tokenizer) = &*PHI;
        let mut pipeline =
            TextGeneration::new(model.clone(), tokenizer.clone(), 398752958, Some(0.3), None, 1.1, 64, &Device::Cpu);
        let response = pipeline.run(&prompt, 400)?;
        Ok(response)
    }
    ```
    *   **`PHI`**: 使用 `lazy_static` 异步加载 `Demonthos/dolphin-2_6-phi-2-candle` 模型。这是一个经过量化的 `Phi-2` 模型，适合在 CPU 上运行。
    *   **`load_model`**: 从 Hugging Face Hub 下载模型权重 (`model-q4k.gguf`) 和分词器，然后加载 `QMixFormer` 模型。
    *   **`TextGeneration`**: 一个用于管理 LLM 推理过程的结构体，包含模型、分词器、LogitsProcessor (用于采样下一个 token)、重复惩罚等参数。
    *   **`run`**: 核心文本生成函数。它接收一个 `prompt` 和 `sample_len` (生成长度)，通过循环迭代，每次生成一个 token，直到达到最大长度或遇到结束 token。过程中会应用重复惩罚。
    *   **`answer_with_context`**: 这是 RAG 的最后一步。它接收用户查询和从数据库检索到的 `references` (相关内容)。它将这些信息格式化为一个特定的提示 (prompt)，然后调用 `TextGeneration` 管道来生成最终答案。提示格式遵循 `phi-2` 模型的 ChatML 格式。

### 4.7 整合所有模块到 `main.rs`

最后，修改 `src/main.rs`，使其能够调用 `database`、`embeddings` 和 `llm` 模块，实现完整的聊天机器人功能。

```rust
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

```

## 5. 运行与测试

1.  **第一次运行 (模型下载)**:
    当首次运行 `cargo run` 或执行 `remember` 或 `ask` 命令时，程序会尝试从 Hugging Face Hub 下载嵌入模型和 LLM 模型。这可能需要一些时间，具体取决于您的网络速度。模型文件通常会缓存到 `~/.cache/huggingface/hub` 目录。
    ```bash
    cargo run -- remember "Rust is a powerful systems programming language focused on safety, performance, and concurrency."
    ```
    第一次执行 `remember` 命令时，会下载 `bge-small-en-v1.5` 嵌入模型。

2.  **存储知识**:
    继续使用 `remember` 命令向本地知识库添加内容。
    ```bash
    cargo run -- remember "Cargo is Rust's package manager and build system."
    cargo run -- remember "Tokio is a runtime for writing asynchronous applications with the Rust programming language."
    ```
    您可以在项目目录下找到一个 `db` 文件夹，其中包含了 `SurrealDB` 存储的数据。

3.  **提问**:
    使用 `ask` 命令向 Chacha 提问。
    ```bash
    cargo run -- ask "What is Rust?"
    cargo run -- ask "What is Cargo?"
    cargo run -- ask "What is Tokio used for?"
    cargo run -- ask "Tell me about systems programming languages."
    ```
    当您首次执行 `ask` 命令时，如果 `llm` 模型尚未下载，它会先进行下载。
    程序会首先根据您的查询从本地知识库中检索相关内容，然后将查询和检索到的内容一起发送给 LLM 进行回答。

    **预期行为**:
    *   程序会输出“Hello, Chacha is starting up!”。
    *   对于 `remember` 命令，会输出“Remembering: ...”和成功插入的 ID。
    *   对于 `ask` 命令，会先输出“Searching for relevant information...”，然后显示检索到的相关内容（如果有），最后输出 LLM 生成的答案。

## 6. 实验过程中遇到的问题及解决方案

*   **`libclang` 缺失**: 在编译 `surrealdb` 时，如果系统没有安装 `clang`，可能会遇到编译错误。
    *   **解决方案**: 根据错误提示安装 `clang`。在 Debian/Ubuntu 系统上，执行 `sudo apt install -y clang`。
*   **编译时间长**: 首次编译 `candle` 和相关模型代码时，编译时间会较长。
    *   **解决方案**: 这是正常现象，耐心等待即可。确保 `Cargo.toml` 中的依赖分支 (`branch = "main"`) 与 `candle` 仓库的最新稳定版本匹配。
*   **内存消耗**: 运行 LLM 可能会消耗大量内存。
    *   **解决方案**: 本实验使用的 `dolphin-2_6-phi-2-candle` 是一个量化模型，相对较小，在一般电脑上运行通常问题不大。如果遇到内存不足，可以尝试在更强大的机器上运行。
*   **异步函数调用**: 确保所有异步操作都在 `async` 函数中，并且使用 `await` 关键字等待结果。
    *   **解决方案**: `#[tokio::main]` 宏帮助处理了 `main` 函数的异步运行时，而内部的 `await` 调用确保了异步操作的正确执行顺序。

## 7. 实验总结

本次实验成功实现了一个基于 Rust 的聊天机器人，集成了命令行交互、本地知识库检索和大型语言模型推理。我们学习了如何：
*   使用 `clap` 构建清晰的命令行界面。
*   利用 `SurrealDB` 结合向量嵌入实现本地知识库的存储与检索。
*   运用 `candle` 框架加载和运行预训练的嵌入模型 (BAAI/bge-small-en-v1.5) 和大型语言模型 (Phi-2)。
*   将检索到的信息作为上下文，增强 LLM 的生成能力，实现 RAG 架构。

通过本次实践，我们对 Rust 生态系统中的异步编程、机器学习推理库以及现代 AI 应用的 RAG 模式有了更深入的理解。该实验为进一步开发更复杂的智能助手奠定了基础。

## 8. 参考文献

*   `lab4-manual.md` - 本次实验的指导手册。
*   `lab4-assignments.md` - 本次实验的作业要求。
*   Hugging Face Candle Repository: [https://github.com/huggingface/candle](https://github.com/huggingface/candle)
*   Hugging Face Hub: [https://huggingface.co/](https://huggingface.co/)
    *   BAAI/bge-small-en-v1.5 model: [https://huggingface.co/BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
    *   Demonthos/dolphin-2_6-phi-2-candle model: [https://huggingface.co/Demonthos/dolphin-2_6-phi-2-candle](https://huggingface.co/Demonthos/dolphin-2_6-phi-2-candle)
*   Clap Documentation: [https://docs.rs/clap/latest/clap/](https://docs.rs/clap/latest/clap/)
*   SurrealDB Documentation: [https://surrealdb.com/docs](https://surrealdb.com/docs)
*   Tokio Documentation: [https://tokio.rs/docs/](https://tokio.rs/docs/)

---