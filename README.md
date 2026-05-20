# zipformer_jp_axera

**日本語 ASR(`reazon-research/japanese-zipformer-base-k2-rs35kh`)を AX650N NPU で動かす変換パイプライン**

PyTorch (HuggingFace) → ONNX → `.axmodel` への完全な変換手順と、AX650N PCIe ボード上で 71.5ms / RTF 0.0071 を実現した実機推論コード一式。

---

## 🎯 結果

| | ONNXRuntime (FP32, CPU) | **axmodel (U16, AX650N NPU)** |
|---|---|---|
| 推論時間 (10 runs mean) | 316.6 ms | **71.5 ms** |
| Real Time Factor | 0.0317 | **0.0071** (= 138× リアルタイム) |
| 認識結果 | こんにちはこれはテキスト読み上げのテストです | (完全一致) |
| 出力 cos sim vs FP32 | — | **0.9879** |
| Speedup | 1.0× | **4.4×** |

入力: 10秒固定 raw waveform `(1, 160000) float32`、出力: CTC logits `(1, 499, 5230)`

---

## 📂 ディレクトリ構成

```
.
├── README.md                       ← このファイル
├── LICENSE                         (MIT)
├── pyproject.toml
├── transcribe_axmodel.py           ← 🎙️ 実機推論 CLI(メイン)
├── benchmark.py                    ← ⏱️ ONNXRuntime vs NPU 比較
│
├── scripts/                        ← 🛠️ 変換パイプライン (順に実行)
│   ├── 00_test_pytorch.py          PyTorch 動作確認
│   ├── 01_export_onnx.py           PyTorch → ONNX (clean export)
│   ├── 02_postprocess_onnx.py      Pow(-0.5) → Sqrt+Div 書換
│   ├── 03_fold_shape_args.py       動的 shape 引数を定数化
│   ├── 04_generate_calib.py        calibration data 生成
│   └── 05_analyze_precision.py     量子化精度 cos 統計
│
└── configs/                        ← ⚙️ pulsar2 build 設定
    ├── encoder_u16.json            FP32 混在 + 高精度 (subgraph 増)
    └── encoder_u16_min.json        全 U16 (fuse 1 subgraph) ★推奨
```

ビルド時の出力(`encoder_ctc*.onnx`, `output_u16/`, `calibrations/`)は再生成可能なので `.gitignore`。

---

## 🚀 クイックスタート

### 必要環境
- Ubuntu 22.04 + AX650N PCIe ボード(driver V3.6.4)
- Docker + `pulsar2:5.2` イメージ
- Python 3.10 + [`uv`](https://github.com/astral-sh/uv)
- `axengine-0.1.3-py3-none-any.whl` (pyaxengine)

### セットアップ

```bash
git clone https://github.com/nnn112358/zipformer_jp_axera.git
cd zipformer_jp_axera
uv sync
uv add /path/to/axengine-0.1.3-py3-none-any.whl
```

### 実行(axmodel 既にある前提)

```bash
# 単一 wav
uv run python transcribe_axmodel.py audio.wav

# ベンチマーク (ONNXRuntime CPU vs axmodel NPU)
uv run python benchmark.py
```

---

## 🛠️ 0 から axmodel を作る(全 7 ステップ)

```bash
# 1. PyTorch 動作確認 (オプション)
uv run python scripts/00_test_pytorch.py

# 2. PyTorch → ONNX (clean export)
uv run python scripts/01_export_onnx.py
#  → encoder_ctc.onnx (~400MB)

# 3. ONNX 最適化 (onnxslim)
uv run python -c "import onnxslim; onnxslim.slim('encoder_ctc.onnx', output_model='encoder_ctc_slim.onnx')"

# 4. Pow(-0.5) 書換
uv run python scripts/02_postprocess_onnx.py
#  → encoder_ctc_final.onnx

# 5. shape 引数定数化
uv run python scripts/03_fold_shape_args.py
#  → encoder_ctc_folded.onnx

# 6. キャリブレーション data 作成
uv run python scripts/04_generate_calib.py
#  → calibrations/input_values.tar.gz

# 7. pulsar2 build (docker)
docker run --rm -v "$(pwd):/data" -w /data pulsar2:5.2 pulsar2 build \
  --input encoder_ctc_folded.onnx \
  --config configs/encoder_u16_min.json \
  --output_dir output_u16 \
  --output_name encoder_u16.axmodel \
  --target_hardware AX650 \
  --compiler.check 0

# root 所有を chown
docker run --rm -v "$(pwd):/data" --entrypoint chown pulsar2:5.2 \
  -R "$(id -u):$(id -g)" /data/output_u16

# 8. 量子化精度確認 (オプション)
uv run python scripts/05_analyze_precision.py

# 9. 実機推論
uv run python transcribe_axmodel.py
```

---

## 🔍 パイプライン概要図

```
   PyTorch (HF transformers, ZipformerForCTC)
        │
        │ 01_export_onnx.py
        │  ├─ Balancer / Whiten         → Identity
        │  ├─ SwooshLForward            → F.softplus(x-4)
        │  ├─ SwooshRForward            → F.softplus(x-1)
        │  ├─ torch.jit.is_tracing      = True (forced)
        │  └─ torch.onnx.export(dynamo=False)
        ▼
   encoder_ctc.onnx                     ← Loop / If が消える
        │
        │ onnxslim
        ▼
   encoder_ctc_slim.onnx
        │
        │ 02_postprocess_onnx.py
        │  └─ Pow(x, -0.5) → Sqrt(x) + Div(1, _)
        ▼
   encoder_ctc_final.onnx
        │
        │ 03_fold_shape_args.py
        │  └─ Reshape/Expand/Slice の shape を ORT で実評価
        ▼
   encoder_ctc_folded.onnx              ← pulsar2 に渡す clean ONNX
        │
        │ pulsar2 build (docker)
        │  + 04_generate_calib.py 製の calibrations/input_values.tar.gz
        │  + configs/encoder_u16_min.json
        ▼
   output_u16/encoder_u16.axmodel       ← fuse 1 subgraph、AXCLRT 互換
        │
        │ transcribe_axmodel.py (pyaxengine)
        ▼
   [認識結果]  "こんにちはこれはテキスト読み上げのテストです"
```

---

## 💡 ハマりどころと解決(全部経験済み)

<details>
<summary><b>1. ONNX に Loop/If が16個ずつ生まれる</b></summary>

**原因**: PyTorch 2.x のデフォルト `torch.onnx.export(... dynamo=True)` では `torch.jit.is_tracing()` が False となり、modeling コードの `as_strided` ブランチが走って Loop に化ける。

**解決**: `dynamo=False` + `unittest.mock.patch("torch.jit.is_tracing", return_value=True)` を併用。
</details>

<details>
<summary><b>2. Log/Exp/Equal/Where が80個ずつ発生</b></summary>

**原因**: scaling.py の `SwooshLForward` / `SwooshRForward` が、tracing 時に `log/exp/where==inf` で safe softplus を手書き実装している。

**解決**: scaling モジュール内のこれら関数を `F.softplus(x - offset) - 0.08*x - bias` に上書き(数学的に等価)。同時に `Balancer` / `Whiten` を no-op に。
</details>

<details>
<summary><b>3. Pow(x, -0.5) が pulsar2 で非サポート</b></summary>

**原因**: `AxPow` は `power ∈ {0.25, 0.5, 2, 4}` のみ対応。負指数不可。`Reciprocal` も量子化非対応。

**解決**: `Pow(x, -0.5)` → `Sqrt(x)` + `Div(1, Sqrt(x))` に書換 (`scripts/02_postprocess_onnx.py`)。
</details>

<details>
<summary><b>4. Reshape/Expand の shape 引数が動的</b></summary>

**原因**: 固定入力なのに `Shape → Gather → Concat` で shape を計算しており、pulsar2 が non-const と判定して落ちる。

**解決**: ORT で実評価して定数化 (`scripts/03_fold_shape_args.py`)。
</details>

<details>
<summary><b>5. axmodel ロード時に board worker が SIGTERM</b></summary>

**原因**: `data_type: "FP32"` を `op_type` で指定する、または NPU 非対応 op が混入すると、subgraph が複数(我々のケースで19個)に分割される。AX650N PCIe runtime は **regular axmodel に 1 subgraph しかサポートしない**。

**解決**: ONNX をクリーンにして `fuse 1 subgraph(s)` を達成。`op_type: FP32` は絶対に使わない。
</details>

<details>
<summary><b>6. 量子化精度が cos 0.30 まで落ちる</b></summary>

**原因**: calibration data が test.wav 1件由来の24サンプル、MinMax 法、silence padding が分布を歪める。

**解決**: `calibration_method: "Percentile"`、`enable_smooth_quant: true`、`ln_scale_data_type: FP32`、padding に振幅 0.001 のホワイトノイズ。最終 cos 0.99 に。
</details>

<details>
<summary><b>7. 10秒固定入力の trailing zero で認識精度落ち</b></summary>

**原因**: attention で全入力を見るので、trailing zero が attention 計算を狂わせて「こん」が脱落するなど。

**解決**: padding に振幅 0.001 のホワイトノイズを混ぜる(silence パディングを避ける)。
</details>

---

## 📊 中間レイヤーの cos 類似度分布

`scripts/05_analyze_precision.py` で `output_u16/quant/debug/precision_analysis_table.txt` を集計した結果(全 1933 レイヤー):

| 指標 | 値 |
|---|---|
| 平均 | 0.9763 |
| 中央値 | 0.9791 |
| 最小 | 0.6607 |
| p25 / p75 / p90 | 0.970 / 0.987 / 0.995 |
| cos < 0.95 のレイヤー | 6.5% |
| cos < 0.9 のレイヤー | 0.8% |
| **負の cos のレイヤー** | **0個** |
| **最終 logits cos** | **0.9833** |

→ 一部の Linear 層は cos 0.66 まで落ちるが、下流で吸収されて最終出力は健全。

---

## 🔧 主要な仕様

| 項目 | 値 |
|---|---|
| モデル | `reazon-research/japanese-zipformer-base-k2-rs35kh` (CTC + Zipformer2) |
| 入力 | raw waveform float32 `(1, 160000)` (= 10秒 × 16kHz) |
| 出力 | CTC logits float32 `(1, 499, 5230)` |
| 量子化 | U16 (全レイヤー) |
| ハードウェア | AX650 (NPU3 mode) |
| axmodel サイズ | 117 MB |
| サブグラフ | **1個** (fully NPU, 0 CPU fallback) |

---

## 📦 依存関係

- `transformers==4.45.*` (新版は modeling_zipformer 互換性で AttributeError 発生)
- `torch==2.12.*` + `torchaudio==2.11.*`
- `onnx`, `onnxslim`, `onnxruntime<1.24` (Python 3.10 のため)
- `librosa`, `soundfile`, `numpy`
- `axengine==0.1.3` (pyaxengine、別途 wheel)
- `pulsar2:5.2` Docker image (10.3 GB)

---

## 📝 License

このリポジトリのコード: MIT

PyTorch モデル `reazon-research/japanese-zipformer-base-k2-rs35kh` の重み: Apache-2.0 (Reazon Holdings Inc.)

---

## 🙏 参考

- [zipformer.axera](https://github.com/AXERA-TECH/zipformer.axera) — `scaling_converter` 手法の参考元
- [icefall (k2-fsa)](https://github.com/k2-fsa/icefall) — Zipformer 本家
- [pulsar2-docs](https://pulsar2-docs.readthedocs.io/) — pulsar2 公式ドキュメント
