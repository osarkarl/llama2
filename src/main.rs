use memmap2::Mmap;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::collections::HashMap;
use std::fs::File;
use std::io::{stdout, BufReader, Read, Write};
use std::mem::size_of;
use std::time::SystemTime;
use std::{env, thread};

// Utilities
trait BytesToNumber {
    fn bytes_to_number(bytes: [u8; 4]) -> Self;
}
impl BytesToNumber for u32 {
    fn bytes_to_number(bytes: [u8; 4]) -> Self {
        u32::from_le_bytes(bytes)
    }
}
impl BytesToNumber for f32 {
    fn bytes_to_number(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
}

fn read_number<T: BytesToNumber>(reader: &mut BufReader<File>) -> T {
    let mut buffer = [0; 4];
    reader.read_exact(&mut buffer).unwrap();
    T::bytes_to_number(buffer)
}

fn read_str(reader: &mut BufReader<File>, length: usize) -> String {
    let mut buffer = vec![0; length];
    reader.read_exact(&mut buffer).unwrap();
    std::str::from_utf8(&buffer).unwrap().to_owned()
}

fn bytes_to_vec<T>(bytes: &[u8], length: usize) -> Vec<T>
where
    T: Clone,
{
    let ptr = bytes.as_ptr() as *const T;
    let slice = unsafe { std::slice::from_raw_parts(ptr, length) };
    return slice.to_vec();
}

fn usage(err: bool) {
    println!("Usage:   cargo run --release <CHECKPOINT> [OPTIONS]");
    println!("Example: cargo run --release stories15M.bin -n 128 -i \"Once upon a time\"");
    println!("Options:");
    println!("  -t <float>  temperature in [0,inf], default 1.0");
    println!("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9");
    println!("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
    println!("  -i <string> input prompt");
    println!();
    if err {
        panic!();
    }
}

// -----------------------------------------------------------------------

// Transformer data structures
#[derive(Debug)]
struct Config {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    n_kv_heads: usize, // number of key/value heads
    vocab_size: usize, // vocabulary size
    seq_len: usize,    // max sequence length
}

impl Config {
    fn read(bytes: &[u8], offset: &mut usize) -> Self {
        let length = 7; // number of elements in Config
        let data: Vec<_> = bytes_to_vec::<u32>(bytes, length)
            .iter()
            .map(|&e| e as usize)
            .collect();
        *offset += size_of::<u32>() * length;

        return Self {
            dim: data[0],
            hidden_dim: data[1],
            n_layers: data[2],
            n_heads: data[3],
            n_kv_heads: data[4],
            vocab_size: data[5],
            seq_len: data[6],
        };
    }
}

#[derive(Debug)]
struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    wq: Vec<f32>, // (layer, dim, n_heads * head_size)
    wk: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    wv: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    wo: Vec<f32>, // (layer, n_heads * head_size, dim)
    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Option<Vec<f32>>, // (vocab_size, dim)
}

impl TransformerWeights {
    fn read(bytes: &[u8], offset: &mut usize, c: &Config) -> Self {
        let head_size = c.dim / c.n_heads;
        let length_map = HashMap::from([
            ("token_embedding_table", c.vocab_size * c.dim),
            ("rms_att_weight", c.n_layers * c.dim),
            ("rms_ffn_weight", c.n_layers * c.dim),
            ("wq", c.n_layers * c.dim * c.dim),
            ("wk", c.n_layers * c.dim * c.n_kv_heads * head_size),
            ("wv", c.n_layers * c.dim * c.n_kv_heads * head_size),
            ("wo", c.n_layers * c.dim * c.dim),
            ("w1", c.n_layers * c.hidden_dim * c.dim),
            ("w2", c.n_layers * c.dim * c.hidden_dim),
            ("w3", c.n_layers * c.dim * c.hidden_dim),
            ("rms_final_weight", c.dim),
            ("freq_cis_real", c.seq_len * c.dim / c.n_heads / 2),
            ("freq_cis_imag", c.seq_len * c.dim / c.n_heads / 2),
            ("wcls", c.vocab_size * c.dim),
        ]);

        let value_size = size_of::<f32>();

        let token_embedding_table =
            bytes_to_vec(&bytes[*offset..], length_map["token_embedding_table"]);
        *offset += value_size * length_map["token_embedding_table"];

        let rms_att_weight = bytes_to_vec(&bytes[*offset..], length_map["rms_att_weight"]);
        *offset += value_size * length_map["rms_att_weight"];

        let wq = bytes_to_vec(&bytes[*offset..], length_map["wq"]);
        *offset += value_size * length_map["wq"];

        let wk = bytes_to_vec(&bytes[*offset..], length_map["wk"]);
        *offset += value_size * length_map["wk"];

        let wv = bytes_to_vec(&bytes[*offset..], length_map["wv"]);
        *offset += value_size * length_map["wv"];

        let wo = bytes_to_vec(&bytes[*offset..], length_map["wo"]);
        *offset += value_size * length_map["wo"];

        let rms_ffn_weight = bytes_to_vec(&bytes[*offset..], length_map["rms_ffn_weight"]);
        *offset += value_size * length_map["rms_ffn_weight"];

        let w1 = bytes_to_vec(&bytes[*offset..], length_map["w1"]);
        *offset += value_size * length_map["w1"];

        let w2 = bytes_to_vec(&bytes[*offset..], length_map["w2"]);
        *offset += value_size * length_map["w2"];

        let w3 = bytes_to_vec(&bytes[*offset..], length_map["w3"]);
        *offset += value_size * length_map["w3"];

        let rms_final_weight = bytes_to_vec(&bytes[*offset..], length_map["rms_final_weight"]);
        *offset += value_size * length_map["rms_final_weight"];

        // skip what used to be freq_cis_real (for RoPE)
        *offset += value_size * length_map["freq_cis_real"];
        //skip what used to be freq_cis_imag (for RoPE)
        *offset += value_size * length_map["freq_cis_imag"];

        let shared_weight = if c.vocab_size > 0 { true } else { false };
        let wcls = if shared_weight {
            None
        } else {
            Some(bytes_to_vec(&bytes[*offset..], length_map["wcls"]))
        };
        *offset += if shared_weight {
            0
        } else {
            value_size * length_map["wcls"]
        };

        return Self {
            token_embedding_table: token_embedding_table,
            rms_att_weight: rms_att_weight,
            rms_ffn_weight: rms_ffn_weight,
            wq: wq,
            wk: wk,
            wv: wv,
            wo: wo,
            w1: w1,
            w2: w2,
            w3: w3,
            rms_final_weight: rms_final_weight,
            wcls: wcls,
        };
    }
}

#[derive(Debug)]
struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    word_to_token: HashMap<String, usize>,
    max_token_length: usize,
}

impl Tokenizer {
    fn read(path: &str, vocab_size: usize) -> Tokenizer {
        let mut tokenizer = Tokenizer {
            vocab: Vec::with_capacity(vocab_size),
            vocab_scores: Vec::with_capacity(vocab_size),
            word_to_token: HashMap::new(),
            max_token_length: 0,
        };

        let file = File::open(path).expect("Cannot open tokenizer.bin!");
        let mut reader = BufReader::new(file);

        tokenizer.max_token_length = read_number::<u32>(&mut reader) as usize;

        for i in 0..vocab_size {
            let score = read_number::<f32>(&mut reader);
            tokenizer.vocab_scores.push(score);

            let string_length = read_number::<u32>(&mut reader) as usize;
            let string = read_str(&mut reader, string_length);
            tokenizer.vocab.push(string.clone());
            tokenizer.word_to_token.insert(string, i);
        }

        return tokenizer;
    }

    fn bpe_encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let mut buffer = String::new();

        for i in text.chars() {
            let token_id = *self.word_to_token.get(&i.to_string()).unwrap();
            tokens.push(token_id);
        }

        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_token_id = usize::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..tokens.len() - 1 {
                buffer.clear();
                buffer.push_str(&self.vocab[tokens[i]]);
                buffer.push_str(&self.vocab[tokens[i + 1]]);
                if let Some(token_id) = self.word_to_token.get(&buffer) {
                    if self.vocab_scores[*token_id] > best_score {
                        best_score = self.vocab_scores[*token_id];
                        best_token_id = *token_id;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            // Merge the best pair and delete the second token
            tokens[best_idx] = best_token_id;
            tokens.remove(best_idx + 1);
        }

        return tokens;
    }
}

struct RunState {
    // current wave of activations
    x: Vec<f32>,      // activation at current time stamp (dim,)
    xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,      // query (dim,)
    k: Vec<f32>,      // key (dim,)
    v: Vec<f32>,      // value (dim,)
    att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits (vocab_size,)
    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

impl RunState {
    fn new(c: &Config) -> Self {
        let kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;
        return Self {
            x: vec![0.0; c.dim],
            xb: vec![0.0; c.dim],
            xb2: vec![0.0; c.dim],
            hb: vec![0.0; c.hidden_dim],
            hb2: vec![0.0; c.hidden_dim],
            q: vec![0.0; c.dim],
            k: vec![0.0; c.dim],
            v: vec![0.0; c.dim],
            att: vec![0.0; c.n_heads * c.seq_len],
            logits: vec![0.0; c.vocab_size],
            key_cache: vec![0.0; c.n_layers * c.seq_len * kv_dim],
            value_cache: vec![0.0; c.n_layers * c.seq_len * kv_dim],
        };
    }
}

struct Transformer {
    config: Config,              // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
    state: RunState,             // buffers for the "wave" of activations in the forward pass
}

impl Transformer {
    fn read(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        let mut offset = 0;
        let config = Config::read(&mmap, &mut offset);
        let weights = TransformerWeights::read(&mmap, &mut offset, &config);
        let state = RunState::new(&config);

        return Self {
            config: config,
            weights: weights,
            state: state,
        };
    }
}

// -----------------------------------------------------------------------

// Neural net operations

fn dot_prod(x: &[f32], y: &[f32]) -> f32 {
    let result = x.iter().zip(y).map(|(&a, &b)| a * b).sum();
    return result;
}

fn mat_mul(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let n = x.len();
    out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
        *out_i = dot_prod(&weight[i * n..(i + 1) * n], &x);
    });
}

fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let mut s = x.iter().map(|&a| a * a).sum::<f32>() / (x.len() as f32);
    s = 1.0 / (s + 1e-5f32).sqrt();

    out.iter_mut()
        .zip(x)
        .zip(weight)
        .for_each(|((o, &x), &w)| *o = w * x * s);
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().reduce(|a, b| a.max(b)).unwrap();
    let mut sum = 0.0;
    x.iter_mut().for_each(|e| {
        *e = (*e - max).exp();
        sum += *e;
    });
    x.iter_mut().for_each(|e| *e /= sum);
}

fn forward(token: usize, pos: usize, transformer: &mut Transformer) {
    let c = &transformer.config;
    let s = &mut transformer.state;
    let w = &transformer.weights;

    let dim = c.dim;
    let hidden_dim = c.hidden_dim;
    let head_size = dim / c.n_heads;
    let seq_len = c.seq_len;

    // copy the token embedding into x
    s.x.copy_from_slice(&w.token_embedding_table[token * dim..(token + 1) * dim]);

    // forward all the layers
    for l in 0..c.n_layers {
        // attention rmsnorm
        rms_norm(&mut s.xb, &s.x, &w.rms_att_weight[l * dim..(l + 1) * dim]);

        // qkv projection
        mat_mul(&mut s.q, &s.xb, &w.wq[l * dim * dim..(l + 1) * dim * dim]);
        mat_mul(&mut s.k, &s.xb, &w.wk[l * dim * dim..(l + 1) * dim * dim]);
        mat_mul(&mut s.v, &s.xb, &w.wv[l * dim * dim..(l + 1) * dim * dim]);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for i in (0..dim).step_by(2) {
            let head_dim = i % head_size;
            let freq = 1.0 / f32::powf(10000.0, head_dim as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = f32::cos(val);
            let fci = f32::sin(val);

            s.q[i] = s.q[i] * fcr - s.q[i + 1] * fci;
            s.q[i + 1] = s.q[i] * fci + s.q[i + 1] * fcr;

            s.k[i] = s.k[i] * fcr - s.k[i + 1] * fci;
            s.k[i + 1] = s.k[i] * fci + s.k[i + 1] * fcr;
        }

        // cache kv values
        let loff = l * seq_len * dim; // layer offset
        s.key_cache[(loff + pos * dim)..(loff + (pos + 1) * dim)].copy_from_slice(&s.k);
        s.value_cache[(loff + pos * dim)..(loff + (pos + 1) * dim)].copy_from_slice(&s.v);

        // multihead attention

        // create groups for each head (for parallel processing)
        let mut atts: Vec<&mut [f32]> = s.att.chunks_mut(seq_len).collect();
        let qs: Vec<&mut [f32]> = s.q.chunks_mut(head_size).collect();
        let xbs: Vec<&mut [f32]> = s.xb.chunks_mut(head_size).collect();

        atts.par_iter_mut()
            .zip(xbs)
            .enumerate()
            .for_each(|(h, (att, xb))| {
                // get the query vector for this head
                let q = &*qs[h];
                // iterate over all timesteps, including the current one
                for t in 0..=pos {
                    // get the key vector for this head and at his timestep
                    let koff = loff + t * dim + h * head_size; // key head offset
                    let k = &s.key_cache[koff..(koff + head_size)];
                    // calculate the attention score
                    att[t] = dot_prod(q, k) / (head_size as f32).sqrt();
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(&mut att[..(pos + 1)]);

                // weighted sum of the values, store back into xb
                xb.fill(0.0);
                for t in 0..=pos {
                    let koff = loff + t * dim + h * head_size; // key head offset
                                                               // get the value vector for this head and at this timestep
                    let v = &s.value_cache[koff..(koff + head_size)];
                    // accumulate the weighted value into xb
                    xb.iter_mut()
                        .zip(v)
                        .for_each(|(xb_i, &v_i)| *xb_i += att[t] * v_i);
                }
            });

        // final matmul to get the output of the attention
        mat_mul(&mut s.xb2, &s.xb, &w.wo[l * dim * dim..(l + 1) * dim * dim]);

        // residual connection
        s.x.iter_mut().zip(s.xb2.iter()).for_each(|(a, b)| *a += *b);

        // ffn rmsnorm
        rms_norm(&mut s.xb, &s.x, &w.rms_ffn_weight[l * dim..(l + 1) * dim]);

        // FFN block: self.w2(F.silu(self.w1(x)) * self.w3(x))
        mat_mul(
            &mut s.hb,
            &s.xb,
            &w.w1[l * hidden_dim * dim..(l + 1) * hidden_dim * dim],
        );
        mat_mul(
            &mut s.hb2,
            &s.xb,
            &w.w3[l * hidden_dim * dim..(l + 1) * hidden_dim * dim],
        );

        // SwiGLU non-linearity
        s.hb.iter_mut()
            .for_each(|e| *e = *e * (1.0 / (1.0 + (-*e).exp())));
        s.hb.iter_mut()
            .zip(s.hb2.iter())
            .for_each(|(a, &b)| *a *= b);

        // final matmul to get the output of the ffn
        mat_mul(
            &mut s.xb,
            &s.hb,
            &w.w2[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
        );

        // residual connection
        s.x.iter_mut().zip(s.xb.iter()).for_each(|(a, &b)| *a += b);
    }

    // final rmsnorm
    s.xb.copy_from_slice(&s.x);
    rms_norm(&mut s.x, &s.xb, &w.rms_final_weight);

    // classifier into logits
    let wcls = match &w.wcls {
        Some(wcls) => wcls,
        None => &w.token_embedding_table,
    };
    mat_mul(&mut s.logits, &s.x, wcls);
}
// -----------------------------------------------------------------------

// Sampling
fn sample_argmax(probs: &[f32]) -> usize {
    // return the index that has the highest probability
    let mut max = probs[0];
    let mut argmax = 0;
    for (i, p) in probs.iter().enumerate() {
        if *p > max {
            max = *p;
            argmax = i;
        }
    }
    return argmax;
}

fn sample_mult(probs: &[f32], coin: f32) -> usize {
    // sample index from probabilities (they must sum to 1!)
    // coin is random number in [0, 1)
    let mut cdf = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cdf += p;
        if coin < cdf {
            return i;
        }
    }
    return probs.len() - 1;
}

fn sample_topp(probs: &[f32], coin: f32, top_p: f32) -> usize {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed probability top_p

    // sort probabilities by decreasing order (and remember position)
    let mut index_prob: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.clone()))
        .collect();
    index_prob.sort_by(|&a, &b| (b.1).partial_cmp(&a.1).unwrap());

    let mut cdf = 0.0;
    let mut cutoff_index = 0;
    for (i, &e) in index_prob.iter().enumerate() {
        cdf += e.1;
        if top_p < cdf {
            break;
        } else {
            cutoff_index = i;
        }
    }

    // sample from the truncated list
    let r = coin * cdf;
    cdf = 0.0;
    for i in 0..=cutoff_index {
        cdf += index_prob[i].1;
        if r < cdf {
            return index_prob[i].0;
        }
    }

    return index_prob[cutoff_index].0;
}

fn sample(logits: &mut [f32], temperature: f32, top_p: f32) -> usize {
    let next;
    if temperature == 0.0 {
        next = sample_argmax(logits);
    } else {
        softmax(logits);

        let coin = rand::thread_rng().gen_range(0.0..1.0);
        if top_p == 0.0 || top_p == 1.0 {
            next = sample_mult(logits, coin);
        } else {
            next = sample_topp(logits, coin, top_p);
        }
    }

    return next;
}

fn generate(
    start_promt: &str,
    steps: usize,
    temperature: f32,
    top_p: f32,
    transformer: &mut Transformer,
    tokenizer: Tokenizer,
) {
    let steps = if steps == 0 || steps > transformer.config.seq_len {
        transformer.config.seq_len
    } else {
        steps
    };

    let mut tokens = Vec::with_capacity(transformer.config.seq_len);
    if start_promt.len() == 0 {
        tokens.push(1);
    } else {
        tokens.append(&mut tokenizer.bpe_encode(start_promt));
    }

    print!("{}{}", tokenizer.vocab[1], &start_promt);
    let mut next_token;
    let mut pos = tokens.len() - 1;
    while pos < steps {
        forward(
            tokens[pos],
            pos,
            transformer,
        );
        next_token = sample(&mut transformer.state.logits, temperature, top_p);

        pos += 1;
        tokens.push(next_token);

        print!("{}", tokenizer.vocab[next_token]);
        stdout().flush().unwrap();
    }
    println!();
}
// -----------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // default parameters
    let mut checkpoint_file = "";
    let mut temperature = 1.0;
    let mut top_p = 0.9;
    let mut steps = 256;
    let mut prompt = "";

    let args: Vec<String> = env::args().collect();
    let args_len = args.len();

    // Argparse so we can override the defaults above from the command line
    if args_len < 2 || args_len % 2 != 0 || args_len > 10 {
        usage(true);
    } else {
        checkpoint_file = &args[1];
        for i in (2..args_len).step_by(2) {
            if args[i] == "-t" {
                temperature = args[i + 1].parse::<f32>()?;
            } else if args[i] == "-p" {
                top_p = args[i + 1].parse::<f32>()?;
            } else if args[i] == "-n" {
                steps = args[i + 1].parse::<usize>()?;
            } else if args[i] == "-i" {
                prompt = &args[i + 1];
            } else {
                usage(true);
            }
        }
    }

    // parameter validation/overrides
    if temperature < 0.0 {
        temperature = 0.0
    };
    if top_p < 0.0 || top_p > 1.0 {
        top_p = 0.9
    };

    let mut transformer = Transformer::read(checkpoint_file);
    let tokenizer = Tokenizer::read("tokenizer.bin", transformer.config.vocab_size);

    // number of threads for execution
    let n_threads = transformer
        .config
        .n_heads
        .min(thread::available_parallelism()?.get());
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()?;

    let start = SystemTime::now();
    generate(
        prompt,
        steps,
        temperature,
        top_p,
        &mut transformer,
        tokenizer,
    );
    let end = start.elapsed().unwrap();
    println!("_______________________________________________");
    println!("Threads: {}", n_threads);
    println!("Time: {}.{} seconds", end.as_secs(), end.subsec_millis());
    println!(
        "Tokens per second: {:0.2}",
        steps as f32 / end.as_secs_f32()
    );

    Ok(())
}
