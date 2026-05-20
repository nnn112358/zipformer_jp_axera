"""PyTorch で japanese-zipformer-base-k2-rs35kh を実行して認識を確認。
test.wav で「こんにちはこれはテキスト読み上げのテストです」が出れば成功。
"""
import sys
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer

REPO = "reazon-research/japanese-zipformer-base-k2-rs35kh"
WAV = sys.argv[1] if len(sys.argv) > 1 else "/home/nnn/Desktop/ReazonSpeech/test.wav"

print(f"Loading {REPO} ...")
feature_extractor = AutoFeatureExtractor.from_pretrained(REPO)
model = AutoModelForCTC.from_pretrained(REPO, trust_remote_code=True)
model.eval()

# Try also tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    print(f"tokenizer: {type(tokenizer).__name__}, vocab_size={tokenizer.vocab_size}")
except Exception as e:
    print(f"tokenizer load failed: {e}")
    tokenizer = None

print(f"\nLoading audio: {WAV}")
audio, sr = librosa.load(WAV, sr=16000)
print(f"  duration: {len(audio)/sr:.2f}s")

inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=sr)
print(f"  feature_extractor outputs keys: {list(inputs.keys())}")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")

with torch.inference_mode():
    outputs = model(**inputs)

print(f"\nmodel outputs type: {type(outputs).__name__}")
print(f"  attributes: {[a for a in dir(outputs) if not a.startswith('_')]}")
if hasattr(outputs, "logits"):
    print(f"  logits shape: {tuple(outputs.logits.shape)} dtype={outputs.logits.dtype}")
elif hasattr(outputs, "last_hidden_state"):
    print(f"  last_hidden_state shape: {tuple(outputs.last_hidden_state.shape)}")

# Decode if logits available
if hasattr(outputs, "logits") and tokenizer is not None:
    pred_ids = outputs.logits.argmax(dim=-1)
    print(f"\npred_ids shape: {tuple(pred_ids.shape)}")
    print(f"raw pred_ids[0][:50]: {pred_ids[0][:50].tolist()}")

    # CTC decode: collapse repeats + remove blanks
    BLANK = 0
    ids = pred_ids[0].tolist()
    collapsed = []
    prev = None
    for i in ids:
        if i != prev and i != BLANK:
            collapsed.append(i)
        prev = i
    text = tokenizer.decode(collapsed, skip_special_tokens=True)
    print(f"\n[result] {text}")
