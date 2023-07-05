#![allow(dead_code)]
// https://github.com/openai/whisper/blob/main/whisper/model.py
// TODO:
// - kv-cache support?
// - language detection?

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use clap::Parser;
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;
mod model;
use model::{Config, VarBuilder, Whisper};

const DTYPE: DType = DType::F32;

// Audio parameters.
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 400;
const N_MELS: usize = 80;
const HOP_LENGTH: usize = 160;
const CHUNK_LENGTH: usize = 30;
const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 480000 samples in a 30-second chunk
const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000 frames in a mel spectrogram input
const N_SAMPLES_PER_TOKEN: usize = HOP_LENGTH * 2; // the initial convolutions has stride 2
const FRAMES_PER_SECOND: usize = SAMPLE_RATE / HOP_LENGTH; // 10ms per audio frame
const TOKENS_PER_SECOND: usize = SAMPLE_RATE / N_SAMPLES_PER_TOKEN; // 20ms per audio token

const NO_SPEECH_THRESHOLD: f64 = 0.6;
const LOGPROB_THRESHOLD: f64 = -1.0;
const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    weights: String,

    #[arg(long)]
    input: String,

    #[arg(long)]
    tokenizer_config: String,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

struct Decode {
    model: Whisper,
    rng: rand::rngs::StdRng,
    tokenizer: Tokenizer,
}

impl Decode {
    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &self.model;
        let audio_features = model.encoder.forward(mel)?;
        println!("audio features: {:?}", audio_features.dims());
        let sample_len = model.config.n_text_ctx / 2;
        let mut sum_logprob = 0f64;
        let no_speech_prob = f64::NAN;
        // TODO: 50257 is the start of transcipt token, be more principled about get initial tokens
        let mut tokens: Vec<u32> = vec![50257];
        for _i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &mel.device())?;
            // Insert a batch dim.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let logits = model.decoder.forward(&tokens_t, &audio_features)?;
            let logits = logits.squeeze(0)?;
            let (seq_len, _) = logits.shape().r2()?;
            let logits = logits.get(seq_len - 1)?;
            let next_token = if t > 0f64 {
                let prs = (&logits / t)?.softmax(logits.rank() - 1)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = logits
                .softmax(logits.rank() - 1)?
                .get(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            sum_logprob += prob.ln();
            // 50256 is the eot token, TODO: parameterize this.
            if next_token == 50256 || tokens.len() > model.config.n_text_ctx {
                break;
            }
        }
        let text = self
            .tokenizer
            .decode(tokens.clone(), true)
            .map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: DecodingResult = self.decode(segment, t)?;
            if i == TEMPERATURES.len() - 1 {
                return Ok(dr);
            }
            let needs_fallback = dr.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                || dr.avg_logprob < LOGPROB_THRESHOLD;
            if !needs_fallback || dr.no_speech_prob > NO_SPEECH_THRESHOLD {
                return Ok(dr);
            }
        }
        unreachable!()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    let rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    let tokenizer = Tokenizer::from_file(args.tokenizer_config).map_err(E::msg)?;

    let input = unsafe { candle::safetensors::MmapedFile::new(args.input)? };
    let input = input.deserialize()?;
    let mel = input.tensor("mel", &device)?;
    println!("loaded mel: {:?}", mel.dims());

    let weights = unsafe { candle::safetensors::MmapedFile::new(args.weights)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, device);
    let model = Whisper::load(&vb, Config::tiny_en())?;
    let mut dc = Decode {
        model,
        rng,
        tokenizer,
    };

    let (_, _, content_frames) = mel.shape().r3()?;
    let mut seek = 0;
    let mut segments = vec![];
    while seek < content_frames {
        let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
        let segment_size = usize::min(content_frames - seek, N_FRAMES);
        let mel_segment = mel.narrow(2, seek, segment_size)?;
        let segment_duration = (segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
        let dr = dc.decode_with_fallback(&mel_segment)?;
        seek += segment_size;
        if dr.no_speech_prob > NO_SPEECH_THRESHOLD && dr.avg_logprob < LOGPROB_THRESHOLD {
            println!("no speech detected, skipping {seek} {dr:?}");
            continue;
        }
        let segment = Segment {
            start: time_offset,
            duration: segment_duration,
            dr,
        };
        println!("{seek} {segment:?}");
        segments.push(segment)
    }
    Ok(())
}