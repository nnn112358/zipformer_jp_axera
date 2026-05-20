# 性能評価

`benchmark.py` で測定した ONNXRuntime vs axmodel の比較と、各サブステップのタイミング分析。

## 1. 結果サマリ

入力: `test.wav` (3.52s 音声、10秒に padding して固定 shape にした float32 `(1, 160000)`)
runs: 10 iterations + 3 warmup
ハードウェア: 同一 PC + AX650N PCIe ボード

| | init (1回) | 推論時間 (10runs mean) | RTF | スピードアップ |
|---|---|---|---|---|
| ONNXRuntime FP32 (CPU) | 539 ms | **316.6 ms** | 0.0317 | 1.0× |
| **axmodel U16 (AX650N NPU)** | **1982 ms** | **71.5 ms** | **0.0071** | **4.4×** |

## 2. タイミングの内訳

`transcribe_axmodel.py --bench` 実行時の breakdown:

```
session init      : 2007 ms   ← axmodel load + PCIe 経由で board に転送
tokenizer init    :  771 ms   ← HuggingFace tokenizer 初期化(初回 download 含む可能性)
audio load        :  887 ms   ← librosa.load(16kHz mono 変換)
pad/preprocess    :    1 ms
NPU inference     :   72 ms   ← axmodel.run() (NPU mcode 実行)
CTC decode        :    4 ms
```

ホットパス(繰り返し呼ぶ部分)は **NPU inference 72ms + CTC decode 4ms ≈ 76ms**。

初期化(session/tokenizer/audio load)は **常駐 service にすれば 1 回限り**なので、定常運用時は無視できる。

## 3. ONNX vs axmodel の出力差

```
input: test.wav padded to 10s, with 0.001 noise tail
                                                                            
[ONNXRuntime FP32]   こんにちはこれはテキスト読み上げのテストです  
[axmodel U16 NPU]    こんにちはこれはテキスト読み上げのテストです  
                                                                            
                     ✅ 完全一致 (match: True)
                     
Output tensor analysis (logits, shape=(1, 499, 5230)):                       
  cos sim            : 0.9879                                                
  max abs diff       : 40.1542  (logits の絶対値は数百単位なので相対誤差 < 5%)
  mean abs diff      : 1.3791
```

## 4. 中間レイヤーの精度

`scripts/05_analyze_precision.py` で `output_u16/quant/debug/precision_analysis_table.txt` を集計:

| 指標 | 値 |
|---|---|
| Total layers analyzed | 1933 |
| Mean cos sim | 0.9763 |
| Median | 0.9791 |
| Min | 0.6607 |
| **Final logits** | **0.9833** |
| Layers cos < 0.95 | 6.5% |
| Layers cos < 0.9 | 0.8% |
| Layers cos < 0.5 | **0%** ← 致命的破綻なし |

詳細は [QUANTIZATION.md](QUANTIZATION.md) の「我々の精度結果」セクション。

## 5. なぜ NPU の方が速いか

`AX650N` NPU の仕様:
- NPU3 mode: 3 NPU core を統合運用
- 演算スループット: 約 18 TOPS (INT8)、約 9 TOPS (U16)
- DDR 帯域: ~17 GB/s
- CMM (NPU 専用メモリ): 7 GB

CPU(典型的な x86 4 core)との比較:
- CPU の演算スループット: ~0.5 TOPS (AVX2 FP32)
- NPU の方が U16 で **約 18 倍の演算能力**

実測の 4.4× speedup はメモリアクセスなどで律速されてる可能性。理論最大値ではない。

## 6. 推論時間の安定性

10 runs での標準偏差:

| | mean | min | max | std |
|---|---|---|---|---|
| ONNXRuntime CPU | 316.6 | 315.3 | 318.6 | **1.2 ms** |
| axmodel NPU | 71.5 | 71.0 | 71.7 | **0.2 ms** |

両方とも非常に安定。NPU の方が更にブレが少ない(専用ハードで他プロセスの影響受けにくい)。

## 7. ベンチマーク再現方法

```bash
cd zipformer_jp_axera
uv run python benchmark.py
```

オプション:

```bash
# warmup / runs 数を変更
uv run python benchmark.py --warmup 5 --runs 30

# 別の wav で
uv run python benchmark.py --wav /path/to/your.wav
```

## 8. 長音声への適用

固定 10秒 axmodel のため、長い音声は chunking が必要:

`transcribe_axmodel.py` の `Transcriber` は内部で 9秒ごとに分割 + 0.5秒オーバーラップ:

```python
def chunk_audio(audio, chunk_sec=9.0, overlap_sec=0.5):
    if len(audio)/SR <= chunk_sec:
        return [audio]
    # ... 9秒チャンクで分割、0.5秒 overlap
```

ただし overlap 部分のマージは簡易実装で、文境界での重複文字が出ることがあり、実用には Voice Activity Detection (VAD) で文区切りを先に検出する方が望ましい。

## 9. 消費電力(推定)

| | TDP | 推論時消費 |
|---|---|---|
| Host PC x86 (8 core) | ~65 W | 20-30 W 増加 |
| AX650N PCIe (NPU3 動作時) | ~10 W | 5-8 W 増加 |

**省電力**: axmodel は CPU 比 **4-6×** 省電力。バッテリ駆動エッジ機器で意味がある差。

## 10. スループット試算

連続バッチ処理時:

| | per-inference | 1時間あたり処理量(10秒 chunk) |
|---|---|---|
| ONNXRuntime CPU | 316.6 ms | 11,370 chunks = **31.6 時間分音声/時間** |
| **axmodel NPU** | 71.5 ms | 50,349 chunks = **140 時間分音声/時間** |

24時間運用なら NPU 1基で **3360時間/日** の音声を処理可能(理論最大値)。

## 関連ドキュメント

- [REPORT.md](REPORT.md) — 全体俯瞰
- [PIPELINE.md](PIPELINE.md) — 変換パイプライン
- [QUANTIZATION.md](QUANTIZATION.md) — 量子化と精度
