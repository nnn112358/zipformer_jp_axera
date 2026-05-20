"""AX650N PCIe 実機で japanese-zipformer-base-k2-rs35kh の axmodel を走らせて日本語ASR。

使い方:
    uv run python transcribe_axmodel.py [audio_file]
    uv run python transcribe_axmodel.py audio1.wav audio2.wav ...
    uv run python transcribe_axmodel.py --bench [audio_file]   # 詳細タイミング計測

モデル:
    output_u16/encoder_u16.axmodel  (U16 量子化、AX650 NPU3 mode, 1 subgraph)
    入力: raw waveform float32 (1, 160000) = 10秒分
    出力: CTC logits (1, 499, 5230)
"""
import argparse
import sys
import time
import numpy as np
import librosa
import axengine as axe
from transformers import AutoTokenizer

REPO = "reazon-research/japanese-zipformer-base-k2-rs35kh"
MODEL = "output_u16/encoder_u16.axmodel"
SAMPLE_RATE = 16000
T_SAMPLES = 10 * SAMPLE_RATE                 # 固定 10 秒入力 (NPU 要件)
CHUNK_SEC = 9.0                              # 重複防止のため 9 秒で chunk
OVERLAP_SEC = 0.5                            # 0.5 秒オーバーラップ
NOISE_FLOOR = 0.001                          # padding 用ノイズ振幅(量子化 0 除算回避)
BLANK_ID = 0


def load_audio(path: str) -> np.ndarray:
    """16kHz mono float32 で読み込み。"""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def pad_to_t_samples(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """T_SAMPLES に padding。末尾には微小ノイズを入れる(量子化対策)。"""
    if len(audio) >= T_SAMPLES:
        return audio[:T_SAMPLES]
    pad = rng.standard_normal(T_SAMPLES - len(audio)).astype(np.float32) * NOISE_FLOOR
    return np.concatenate([audio, pad])


def ctc_collapse(ids: list[int], blank: int = BLANK_ID) -> list[int]:
    """CTC greedy decode: 連続重複削除 + blank 除去。"""
    out: list[int] = []
    prev = None
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def chunk_audio(audio: np.ndarray, chunk_sec: float = CHUNK_SEC,
                overlap_sec: float = OVERLAP_SEC) -> list[np.ndarray]:
    """長い音声を chunk_sec ごとに区切る(overlap_sec で重複)。"""
    audio_sec = len(audio) / SAMPLE_RATE
    if audio_sec <= chunk_sec:
        return [audio]
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    stride = chunk_samples - int(overlap_sec * SAMPLE_RATE)
    chunks = []
    pos = 0
    while pos < len(audio):
        chunks.append(audio[pos: pos + chunk_samples])
        pos += stride
    return chunks


def merge_chunk_results(chunk_texts: list[str]) -> str:
    """単純結合 (overlap 領域の重複は CTC collapse で大体吸収される)。"""
    # 簡易: 各 chunk text 内で末尾 N 文字と次 chunk 先頭 N 文字を比較し重複を除く
    if not chunk_texts:
        return ""
    out = chunk_texts[0]
    for next_text in chunk_texts[1:]:
        # overlap で同じ部分が両方の chunk に出る可能性 → 簡易マージ
        overlap = ""
        for n in range(min(len(out), len(next_text)), 0, -1):
            if out.endswith(next_text[:n]):
                overlap = next_text[:n]
                break
        out += next_text[len(overlap):]
    return out


class Transcriber:
    def __init__(self, model_path: str = MODEL, repo: str = REPO):
        self._t = {}
        t0 = time.perf_counter()
        self.session = axe.InferenceSession(model_path)
        self._t["session_init_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self._t["tokenizer_init_ms"] = (time.perf_counter() - t0) * 1000

        self.rng = np.random.default_rng(0)

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> tuple[str, dict]:
        """単一チャンク (≤10秒) を処理。timing dict 付き。"""
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        x = pad_to_t_samples(audio_chunk, self.rng)
        t["pad_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        logits = self.session.run(None, {"input_values": x[None].astype(np.float32)})[0]
        t["inference_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ids = logits[0].argmax(-1).tolist()
        collapsed = ctc_collapse(ids)
        text = self.tokenizer.decode(collapsed, skip_special_tokens=True)
        t["decode_ms"] = (time.perf_counter() - t0) * 1000
        return text, t

    def transcribe(self, audio_path: str, verbose: bool = False) -> tuple[str, dict]:
        """ファイル全体を処理(必要なら chunk 分割)。"""
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        audio = load_audio(audio_path)
        t["audio_load_ms"] = (time.perf_counter() - t0) * 1000
        audio_sec = len(audio) / SAMPLE_RATE

        chunks = chunk_audio(audio)
        texts = []
        chunk_timings = []
        for i, chunk in enumerate(chunks):
            text, t_chunk = self.transcribe_chunk(chunk)
            texts.append(text)
            chunk_timings.append(t_chunk)
            if verbose and len(chunks) > 1:
                print(f"  [chunk {i+1}/{len(chunks)}] {len(chunk)/SAMPLE_RATE:.2f}s  "
                      f"infer={t_chunk['inference_ms']:.1f}ms  text=\"{text}\"")
        full_text = merge_chunk_results(texts)
        t["audio_sec"] = audio_sec
        t["n_chunks"] = len(chunks)
        t["chunks"] = chunk_timings
        t["inference_ms_total"] = sum(c["inference_ms"] for c in chunk_timings)
        t["rtf"] = t["inference_ms_total"] / 1000 / audio_sec if audio_sec else float("nan")
        return full_text, t


def print_timing(name: str, t: dict) -> None:
    print(f"\n=== {name} ===")
    if "audio_load_ms" in t:
        print(f"  audio load           : {t['audio_load_ms']:8.1f} ms")
        print(f"  audio length         : {t['audio_sec']:8.2f} s  ({t['n_chunks']} chunk{'s' if t['n_chunks']>1 else ''})")
    if "inference_ms_total" in t:
        print(f"  NPU inference total  : {t['inference_ms_total']:8.1f} ms")
        if t.get("n_chunks", 1) > 1:
            print(f"  NPU inference / chunk: {t['inference_ms_total']/t['n_chunks']:8.1f} ms")
        print(f"  RTF (inference only) : {t['rtf']:8.4f}  (= {1/t['rtf']:.0f}x faster than real-time)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="*", help="音声ファイル (省略時は test.wav)")
    ap.add_argument("--bench", action="store_true", help="benchmark mode (詳細タイミング)")
    ap.add_argument("--model", default=MODEL, help=f"axmodel path (default: {MODEL})")
    args = ap.parse_args()

    audio_files = args.audio if args.audio else ["/home/nnn/Desktop/ReazonSpeech/test.wav"]

    print(f"Loading model: {args.model}")
    asr = Transcriber(args.model)
    print(f"  session init    : {asr._t['session_init_ms']:8.1f} ms")
    print(f"  tokenizer init  : {asr._t['tokenizer_init_ms']:8.1f} ms")

    for path in audio_files:
        print(f"\n>>> {path}")
        try:
            text, t = asr.transcribe(path, verbose=args.bench)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print(f"  [result] {text}")
        print_timing(path, t)


if __name__ == "__main__":
    main()
