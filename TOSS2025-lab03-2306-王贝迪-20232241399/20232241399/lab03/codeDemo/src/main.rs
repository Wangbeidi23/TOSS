use std::io::{self, Write};
use std::process::{Command, Stdio, ChildStdout};
use std::env;

/// 主函数：Shell 的主循环
fn main() -> io::Result<()> {
    // 禁用 Ctrl+C 信号处理，以便子进程能够正常接收信号。
    // 注意：这不是一个生产级 Shell 的完整信号处理方案，
    // 生产级 Shell 会更复杂地管理进程组和信号。
    // For this simple example, we let Ctrl+C kill the shell and its children.
    // If we wanted to ignore it for the shell itself, we would need signal handling crates.

    loop {
        // 打印提示符
        print!("shell> ");
        io::stdout().flush()?; // 确保提示符立即显示

        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?; // 读取用户输入

        // 处理 EOF (Ctrl+D)
        if bytes_read == 0 {
            println!("\nExiting shell.");
            break;
        }

        let input = input.trim(); // 移除首尾空白

        if input.is_empty() {
            continue; // 空输入，继续循环
        }

        // 解析输入，支持管道和引号
        match parse_input(input) {
            Ok(pipeline) => {
                if pipeline.is_empty() || pipeline.iter().any(|cmd| cmd.is_empty()) {
                    eprintln!("shell: invalid command or empty command in pipeline");
                    continue;
                }

                // 执行命令管道
                if let Err(e) = execute_command_pipeline(pipeline) {
                    // 错误信息已经在 execute_command_pipeline 中打印，这里简单处理
                    eprintln!("shell: command execution failed: {}", e);
                }
            }
            Err(e) => {
                eprintln!("shell: parse error: {}", e);
            }
        }
    }

    Ok(())
}

/// 解析输入字符串，支持带引号的参数和管道符
/// 返回一个 Vec<Vec<String>>，外层 Vec 表示管道中的命令，内层 Vec 表示单个命令的参数
fn parse_input(input: &str) -> io::Result<Vec<Vec<String>>> {
    let mut pipeline_commands: Vec<Vec<String>> = Vec::new();
    let mut current_command_str = String::new();
    let mut in_quote = false; // 标记是否在双引号内

    // 遍历输入字符串，先按管道符分割
    for char_code_point in input.chars() {
        if char_code_point == '"' {
            in_quote = !in_quote; // 切换引号状态
            current_command_str.push(char_code_point); // 保留引号在字符串中，后面再处理
        } else if char_code_point == '|' && !in_quote {
            // 遇到管道符且不在引号内，则当前命令字符串结束
            let args = parse_arguments(&current_command_str)?;
            if args.is_empty() {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "Empty command part in pipeline"));
            }
            pipeline_commands.push(args);
            current_command_str.clear(); // 清空，准备下一个命令
        } else {
            current_command_str.push(char_code_point);
        }
    }

    // 处理最后一个命令字符串
    if !current_command_str.is_empty() {
        let args = parse_arguments(&current_command_str)?;
        if args.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Empty command part at end of pipeline"));
        }
        pipeline_commands.push(args);
    }

    if in_quote {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "Unmatched quote"));
    }

    Ok(pipeline_commands)
}

/// 解析单个命令字符串中的参数，支持带引号的参数
fn parse_arguments(command_str: &str) -> io::Result<Vec<String>> {
    let mut args = Vec::new();
    let mut current_arg = String::new();
    let mut in_double_quote = false;
    let mut in_single_quote = false;
    let mut chars = command_str.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' => {
                if in_single_quote {
                    current_arg.push(c);
                } else {
                    in_double_quote = !in_double_quote;
                }
            },
            '\'' => {
                if in_double_quote {
                    current_arg.push(c);
                } else {
                    in_single_quote = !in_single_quote;
                }
            },
            ' ' | '\t' => {
                if in_double_quote || in_single_quote {
                    current_arg.push(c); // 在引号内，空格是参数的一部分
                } else {
                    // 不在引号内，遇到空白则当前参数结束
                    if !current_arg.is_empty() {
                        args.push(current_arg.clone());
                        current_arg.clear();
                    }
                }
            },
            '\\' => { // 简单处理转义，只处理 \" \' \\
                if let Some(next_c) = chars.peek() {
                    match next_c {
                        '"' | '\'' | '\\' => {
                            current_arg.push(*next_c);
                            chars.next(); // Consume the escaped character
                        },
                        _ => current_arg.push(c), // Not a recognized escape, push backslash
                    }
                } else {
                    current_arg.push(c); // Backslash at end of string
                }
            },
            _ => {
                current_arg.push(c);
            }
        }
    }

    // 添加最后一个参数（如果存在）
    if !current_arg.is_empty() {
        args.push(current_arg);
    }

    if in_double_quote || in_single_quote {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "Unmatched quote in argument"));
    }

    Ok(args)
}

/// 执行命令管道
fn execute_command_pipeline(pipeline: Vec<Vec<String>>) -> io::Result<()> {
    let mut previous_stdout: Option<ChildStdout> = None; // 用于连接管道的输出和输入
    let mut children_processes = Vec::new(); // 存储所有子进程，以便后续等待它们完成

    for (i, args) in pipeline.clone().into_iter().enumerate() { // Clone pipeline to avoid move
        if args.is_empty() {
            // 理论上 parse_input 已经检查了，这里做二次防御
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Empty command in pipeline detected during execution"));
        }

        let command_name = &args[0];
        let command_args = &args[1..];

        // 特殊处理内置命令 (如 'exit', 'cd')
        // 注意：内置命令只能在管道的第一个位置起作用，因为它们改变的是 Shell 自身的状态
        if i == 0 {
            if command_name == "exit" {
                // 如果是 'exit' 命令，则直接退出程序
                // 在 main 函数中已经处理了 Ctrl+D 的退出
                std::process::exit(0);
            } else if command_name == "cd" {
                // 'cd' 命令：改变当前工作目录
                if command_args.is_empty() {
                    // cd 没有参数，通常回到 HOME 目录
                    let home_dir = env::var("HOME").unwrap_or_else(|_| "/".to_string());
                    if let Err(e) = env::set_current_dir(&home_dir) {
                        eprintln!("cd: failed to change directory to {}: {}", home_dir, e);
                    }
                } else if command_args.len() == 1 {
                    // cd 后面跟一个路径
                    if let Err(e) = env::set_current_dir(&command_args[0]) {
                        eprintln!("cd: no such file or directory: {}", command_args[0]);
                    }
                } else {
                    eprintln!("cd: too many arguments");
                }
                // 'cd' 是内置命令，不启动外部进程，处理完毕后直接返回
                return Ok(());
            }
        }

        let mut command = Command::new(command_name);
        command.args(command_args);

        // 如果存在上一个命令的标准输出，则将其连接到当前命令的标准输入
        if let Some(stdin) = previous_stdout.take() { // `take()` 用于获取 Option 内部的值并将其设置为 None
            command.stdin(stdin);
        }

        // 如果不是管道中的最后一个命令，则将其标准输出重定向到管道
        if i < pipeline.len() - 1 { // pipeline.len() is now valid due to clone
            command.stdout(Stdio::piped());
        }

        // 启动子进程
        match command.spawn() {
            Ok(mut child) => { // child needs to be mutable to call .take()
                previous_stdout = child.stdout.take(); // Use take() to avoid partial move
                children_processes.push(child); // 将子进程句柄添加到列表中
            }
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    eprintln!("shell: command not found: {}", command_name);
                } else {
                    eprintln!("shell: error executing command {}: {}", command_name, e);
                }
                // 如果启动失败，清除之前的管道输出，避免后续命令尝试连接到无效的句柄
                previous_stdout = None;
                return Err(e); // 向上层传播错误
            }
        }
    }

    // 等待所有管道中的子进程完成
    for mut child in children_processes {
        let status = child.wait()?; // 等待子进程完成并获取其退出状态
        if !status.success() {
            eprintln!("shell: command exited with non-zero status: {:?}", status.code());
        }
    }

    Ok(())
}