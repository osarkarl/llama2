## llama2

This is a Rust port of [Llama2 inference on CPU](https://github.com/karpathy/llama2.c).

#### How to run?
1. Get one of these models:

    [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin)\
    [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin)\
    [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin)
    
    Details of the above models are in the [model section of the original repo](https://github.com/karpathy/llama2.c?tab=readme-ov-file#models).

	`tokenizer.bin` in the repo is required also.
2. Compile and run the Rust code:
    ```
    cargo run --release <CHECKPOINT> [OPTIONS]
    ```
    Example:
    ```
    cargo run --release stories110M.bin -n 128 -i "Once upon a time"
    ```

    Command line options:\
    `-t <float> - temperature in [0,inf], default 1.0`\
    `-p <float> - p value in top-p (nucleus) sampling in [0,1], default 0.9`\
    `-n <int> - number of steps to run for, default 256. 0 = max_seq_len`\
    `-i <string> - input prompt, default ""`
