"""ONNXRuntime (CPU FP32) vs axmodel (AX650N NPU U16) の正式ベンチマーク。

入力: raw waveform float32 (1, 160000)
比較: 認識結果一致 + 推論時間 + RTF + cosine sim
"""
import time
import argparse
import numpy as np
import librosa
import onnxruntime as ort
import axengine as axe
from transformers import AutoTokenizer

REPO = "reazon-research/japanese-zipformer-base-k2-rs35kh"
ONNX = "encoder_ctc_folded.onnx"
AXMODEL = "output_u16/encoder_u16.axmodel"
WAV = "/home/nnn/Desktop/ReazonSpeech/test.wav"
T_SAMPLES = 10 * 16000


def prepare_input(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=16000)
    rng = np.random.default_rng(0)
    if len(audio) < T_SAMPLES:
        pad = rng.standard_normal(T_SAMPLES - len(audio)).astype(np.float32) * 0.001
        audio = np.concatenate([audio, pad])
    else:
        audio = audio[:T_SAMPLES]
    return audio[None].astype(np.float32)


def ctc_decode(logits: np.ndarray, tokenizer) -> str:
    ids = logits[0].argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != 0:
            out.append(i)
        prev = i
    return tokenizer.decode(out, skip_special_tokens=True)


def bench_runner(name: str, init_fn, run_fn, x, n_warmup=3, n_runs=10):
    print(f"\n=== {name} ===")
    t0 = time.perf_counter()
    sess = init_fn()
    t_init = (time.perf_counter() - t0) * 1000
    print(f"  init      : {t_init:8.1f} ms")

    # warmup
    for _ in range(n_warmup):
        run_fn(sess, x)
    # measure
    ts = []
    last_logits = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        last_logits = run_fn(sess, x)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    print(f"  inference : mean={ts.mean():8.1f} ms  min={ts.min():.1f}  max={ts.max():.1f}  std={ts.std():.1f}  ({n_runs} runs)")
    return t_init, ts.mean(), last_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=WAV)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    print(f"Loading: {args.wav}")
    x = prepare_input(args.wav)
    print(f"  input: {x.shape} {x.dtype}  ({x.shape[1]/16000:.2f}s)")

    tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)

    # ONNXRuntime (FP32 CPU, optimization OFF since ORT has issue with our model)
    def init_ort():
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(ONNX, sess_options=opts, providers=["CPUExecutionProvider"])

    def run_ort(sess, x):
        return sess.run(None, {"input_values": x})[0]

    t_ort_init, t_ort_inf, logits_ort = bench_runner(
        "ONNXRuntime (FP32, CPU)", init_ort, run_ort, x,
        n_warmup=args.warmup, n_runs=args.runs)

    # axmodel (U16 NPU)
    def init_ax():
        return axe.InferenceSession(AXMODEL)

    def run_ax(sess, x):
        return sess.run(None, {"input_values": x})[0]

    t_ax_init, t_ax_inf, logits_ax = bench_runner(
        "axmodel (U16, AX650N NPU)", init_ax, run_ax, x,
        n_warmup=args.warmup, n_runs=args.runs)

    # Compare outputs
    print("\n=== Output comparison (FP32 vs U16) ===")
    if logits_ort.shape == logits_ax.shape:
        diff = logits_ort - logits_ax
        cos = (logits_ort.flatten() @ logits_ax.flatten()) / (
            np.linalg.norm(logits_ort) * np.linalg.norm(logits_ax))
        print(f"  shape       : {logits_ort.shape}")
        print(f"  cos sim     : {cos:.4f}")
        print(f"  max abs diff: {np.abs(diff).max():.4f}")
        print(f"  mean abs diff: {np.abs(diff).mean():.4f}")
    else:
        print(f"  shape mismatch: ort={logits_ort.shape}  ax={logits_ax.shape}")

    # Decode both
    txt_ort = ctc_decode(logits_ort, tokenizer)
    txt_ax = ctc_decode(logits_ax, tokenizer)
    print(f"\n  [ORT fp32]   {txt_ort}")
    print(f"  [axmodel u16] {txt_ax}")
    print(f"  match: {txt_ort == txt_ax}")

    # Summary
    audio_sec = x.shape[1] / 16000
    print("\n=== Summary ===")
    print(f"  {'':<30} {'init (ms)':>12} {'infer (ms)':>12} {'RTF':>10}  {'speedup':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}  {'-'*10}")
    print(f"  {'ONNXRuntime FP32 (CPU)':<30} {t_ort_init:>12.1f} {t_ort_inf:>12.1f} {t_ort_inf/1000/audio_sec:>10.4f}  {'1.0x':>10}")
    print(f"  {'axmodel U16 (AX650N NPU)':<30} {t_ax_init:>12.1f} {t_ax_inf:>12.1f} {t_ax_inf/1000/audio_sec:>10.4f}  {t_ort_inf/t_ax_inf:>9.1f}x")


if __name__ == "__main__":
    main()
