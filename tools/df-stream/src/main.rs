//! DeepFilterNet3 streaming shim: read mono f32le @48k on stdin, write enhanced on stdout.
//! Processes one hop (480 samples, 10 ms) per iteration with the model's own ~2-frame
//! lookahead, so end-to-end delay is ~40 ms. Same model + runtime as the deep-filter CLI.
use df::tract::*;
use ndarray::Array2;
use std::io::{self, Read, Write};

fn main() {
    // DF_ATTEN_LIM (dB): cap on how much the model may remove, mixing the input back in.
    // Unlimited (100) eats quiet reverberant speech; 12 keeps a real voice intact.
    let atten: f32 = std::env::var("DF_ATTEN_LIM").ok().and_then(|v| v.parse().ok()).unwrap_or(100.0);
    let r_params = RuntimeParams::default_with_ch(1).with_atten_lim(atten);
    let mut df = DfTract::new(DfParams::default(), &r_params).expect("init DeepFilterNet");
    let (sr, hop) = (df.sr, df.hop_size);
    eprintln!("df-stream ready sr={sr} hop={hop} atten_lim={atten} dB");
    let (stdin, stdout) = (io::stdin(), io::stdout());
    let (mut inp, mut out) = (stdin.lock(), stdout.lock());
    let mut buf = vec![0u8; hop * 4];
    let mut inframe = Array2::<f32>::zeros((1, hop));
    let mut outframe = Array2::<f32>::zeros((1, hop));
    let mut obuf = vec![0u8; hop * 4];
    while inp.read_exact(&mut buf).is_ok() {
        for (i, c) in buf.chunks_exact(4).enumerate() {
            inframe[[0, i]] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        df.process(inframe.view(), outframe.view_mut()).expect("df process");
        for (i, v) in outframe.iter().enumerate() {
            obuf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        if out.write_all(&obuf).is_err() || out.flush().is_err() {
            break;
        }
    }
}
