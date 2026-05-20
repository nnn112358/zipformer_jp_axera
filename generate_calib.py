"""calibration 用 .tar.gz を生成。

入力は raw waveform (1, T=160000) float32 一発のみなので、
test.wav の様々な perturbation を 30 件作る。
"""
import os
import glob
import tarfile
import numpy as np
import soundfile as sf
import librosa

SRC_WAV = "/home/nnn/Desktop/ReazonSpeech/test.wav"
T_FIXED = 10 * 16000
OUT_DIR = "calibrations"
N_SAMPLES = 30


def load_wav(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


def pad_or_trim(wav, T, rng):
    """T サンプルに padding/trim。末尾に微小ノイズ padding(無音 = activation 0 を避ける)。"""
    if len(wav) < T:
        pad = rng.standard_normal(T - len(wav)).astype(np.float32) * 0.001
        return np.concatenate([wav, pad])
    return wav[:T]


def main():
    os.makedirs(f"{OUT_DIR}/input_values", exist_ok=True)
    wav = load_wav(SRC_WAV)
    rng = np.random.default_rng(0)

    print(f"Loaded test.wav: {len(wav)/16000:.2f}s, mean={wav.mean():.4f}, std={wav.std():.4f}")

    variants = []
    # 1) original (with 0/100/300/500/800 ms head silence padding)
    for ms in [0, 100, 200, 300, 500, 800]:
        head = np.zeros(int(16000 * ms / 1000), dtype=np.float32)
        v = np.concatenate([head, wav])
        variants.append((f"orig_head{ms}ms", v))
    # 2) gain variants
    for g in [0.3, 0.5, 0.8, 1.2, 1.5, 2.0]:
        v = np.clip(wav * g, -1, 1).astype(np.float32)
        variants.append((f"gain{g}", v))
    # 3) noise added
    for sigma in [0.001, 0.003, 0.01]:
        n = rng.standard_normal(len(wav)).astype(np.float32) * sigma
        variants.append((f"noisy_{sigma}", wav + n))
    # 4) audio centered with zero pads
    for ms in [200, 500, 1000]:
        head = np.zeros(int(16000 * ms / 1000), dtype=np.float32)
        tail = np.zeros(int(16000 * ms / 1000), dtype=np.float32)
        variants.append((f"center_pad{ms}", np.concatenate([head, wav, tail])))
    # 5) audio offset at different positions
    for ms in [500, 1500, 3000, 5000]:
        head = np.zeros(int(16000 * ms / 1000), dtype=np.float32)
        variants.append((f"offset{ms}ms", np.concatenate([head, wav])))
    # 6) audio chunks (cropped)
    L = len(wav)
    variants.append(("head_half", wav[:L//2]))
    variants.append(("mid_half", wav[L//4:3*L//4]))
    variants.append(("tail_half", wav[L//2:]))
    variants.append(("first_quarter", wav[:L//4]))
    # 7) random noise (silence-ish)
    for sigma in [0.01, 0.05]:
        variants.append((f"pure_noise{sigma}", rng.standard_normal(L).astype(np.float32) * sigma))

    variants = variants[:N_SAMPLES]
    print(f"Generating {len(variants)} samples ...")

    for name, v in variants:
        x = pad_or_trim(v, T_FIXED, rng)[None].astype(np.float32)  # (1, 160000)
        np.save(f"{OUT_DIR}/input_values/{name}.npy", x)
        print(f"  {name:25s}: shape={x.shape} mean={x.mean():.4f} std={x.std():.4f}")

    # tar.gz
    tar_path = f"{OUT_DIR}/input_values.tar.gz"
    with tarfile.open(tar_path, "w:gz") as t:
        for f in sorted(glob.glob(f"{OUT_DIR}/input_values/*.npy")):
            t.add(f, arcname=os.path.basename(f))
    print(f"\nWrote {tar_path}")


if __name__ == "__main__":
    main()
