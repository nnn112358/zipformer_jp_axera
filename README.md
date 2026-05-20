# japanese-zipformer-base-k2-rs35kh → AX650N axmodel

`reazon-research/japanese-zipformer-base-k2-rs35kh` (CTC 日本語 ASR、PyTorch) を AX650N NPU 向け `.axmodel` に変換し、AXCLRT PCIe ボード上で実機推論するパイプライン。

## 結果

| 項目 | ONNXRuntime (FP32, CPU) | **axmodel (U16, AX650N NPU)** |
|---|---|---|
| 推論時間 | 316.6 ms | **71.5 ms** (4.4× faster) |
| RTF | 0.0317 | **0.0071** (138× リアルタイム) |
| 認識結果 | こんにちはこれはテキスト読み上げのテストです | (同上) |
| 量子化精度 (cos sim) | - | 0.9879 |

入力: 10秒固定の raw waveform `(1, 160000) float32`。出力: CTC logits `(1, 499, 5230)`。

## パイプライン全体図

```
PyTorch (HF transformers)
    │
    │ export_onnx.py
    │  ├─ Balancer / Whiten → Identity 化
    │  ├─ SwooshLForward / SwooshRForward → F.softplus ベースに置換
    │  ├─ torch.jit.is_tracing → True 固定 (dynamo=False)
    │  └─ torch.onnx.export
    ▼
encoder_ctc.onnx
    │
    │ onnxslim
    ▼
encoder_ctc_slim.onnx
    │
    │ postprocess_onnx.py
    │  └─ Pow(x, -0.5) → Sqrt(x) → Div(1, _)
    ▼
encoder_ctc_final.onnx
    │
    │ fold_shape_args.py
    │  └─ Reshape/Expand/Slice の shape引数を ORT 実評価で定数化
    ▼
encoder_ctc_folded.onnx  ← pulsar2 build に渡す clean ONNX
    │
    │ pulsar2:5.2 build (docker)
    │  config_encoder_u16_min.json
    │  └─ U16 + Percentile + smooth quant + ln_scale FP32
    ▼
output_u16/encoder_u16.axmodel  ← fuse 1 subgraph、AXCLRT 互換
    │
    │ transcribe_axmodel.py (pyaxengine)
    ▼
[認識結果]
```

## ハマりどころと解決(全部経験済み)

### 1. 元 ONNX に `Loop` / `If` が16個ずつ生まれる

**原因**: PyTorch 2.x の `torch.onnx.export(... dynamo=True)` がデフォルト動作で、`torch.jit.is_tracing()` が False を返すので modeling コードの `as_strided` ブランチが走り Loop に化ける。

**解決**: `dynamo=False` + `unittest.mock.patch("torch.jit.is_tracing", return_value=True)`。

### 2. `Log/Exp/Equal/Where` が各80個発生

**原因**: scaling.py の `SwooshLForward`/`SwooshRForward` 関数が、tracing 時に `log/exp` + `where==inf` で safe softplus を手書き実装。

**解決**: scaling モジュール内の関数を `F.softplus(x - offset) - 0.08*x - bias` で上書き(数学的に等価)。同時に `Balancer`/`Whiten` の forward を no-op に。

### 3. `Pow(x, -0.5)` が pulsar2 で非サポート

**原因**: AxPow は `power=0.25/0.5/2/4` のみ対応、負指数不可。`Reciprocal` も量子化非対応。

**解決**: `Pow(x, -0.5)` → `Sqrt(x)` + `Div(1, Sqrt(x))` に書き換え(postprocess_onnx.py)。

### 4. `Reshape/Expand` の shape 引数が動的

**原因**: 固定入力なのに `Shape→Gather→Concat` で shape が計算され、pulsar2 が non-const と判定して落ちる。

**解決**: ORT で実評価して定数化 (fold_shape_args.py)。

### 5. axmodel ロード時に board worker が SIGTERM

**原因**: `data_type: "FP32"` の `op_type` 指定や、NPU 非対応 op (Log/Exp/Where 等) の混入により subgraph が 19 個に分割される。AX650N PCIe runtime は単一 subgraph axmodel しかサポートしない(LLM-format は別 path)。

**解決**: ONNX を clean にして `fuse 1 subgraph(s)` を達成。`op_type: FP32` は絶対に使わない。

### 6. 量子化精度が cos 0.30 まで落ちる

**原因**: calibration data 1件(test.wav)由来の 24サンプル、MinMax 法。silence padding が分布を歪める。

**解決**: padding に微小ノイズ(0.001振幅)、`calibration_method: Percentile`、`enable_smooth_quant: true`、`ln_scale_data_type: FP32`。最終 cos 0.99 に。

### 7. 10秒固定入力の trailing zero で認識精度落ち

**原因**: モデルは attention で全入力を見るので、trailing zero が attention 計算を狂わせる。

**解決**: padding に振幅 0.001 のホワイトノイズを入れる。

## ファイル一覧

| ファイル | 用途 |
|---|---|
| `export_onnx.py` | PyTorch → ONNX (Balancer/Whiten/Swoosh patch) |
| `postprocess_onnx.py` | Pow(-0.5) → Sqrt + Div(1,_) 書き換え |
| `fold_shape_args.py` | shape 引数の ORT 評価定数化 |
| `generate_calib.py` | calibration data 生成 (24 perturbation) |
| `config_encoder_u16_min.json` | pulsar2 build 設定 (全 U16) |
| `transcribe_axmodel.py` | 実機推論 wrapper (CTC decode + chunking) |
| `benchmark.py` | ORT vs axmodel ベンチマーク |

## 再現手順

```bash
# 1. 依存関係
uv sync

# 2. PyTorch → ONNX
uv run python export_onnx.py

# 3. ONNX 後処理
uv run python -c "import onnxslim; onnxslim.slim('encoder_ctc.onnx', output_model='encoder_ctc_slim.onnx')"
uv run python postprocess_onnx.py encoder_ctc_slim.onnx encoder_ctc_final.onnx
uv run python fold_shape_args.py encoder_ctc_final.onnx encoder_ctc_folded.onnx

# 4. calibration data
uv run python generate_calib.py

# 5. pulsar2 build (docker)
docker run --rm -v "$(pwd):/data" -w /data pulsar2:5.2 pulsar2 build \
  --input encoder_ctc_folded.onnx \
  --config config_encoder_u16_min.json \
  --output_dir output_u16 \
  --output_name encoder_u16.axmodel \
  --target_hardware AX650 \
  --compiler.check 0
docker run --rm -v "$(pwd):/data" --entrypoint chown pulsar2:5.2 -R "$(id -u):$(id -g)" /data/output_u16

# 6. 実機推論
uv run python transcribe_axmodel.py [audio.wav]

# 7. ベンチマーク
uv run python benchmark.py
```

## ライセンス

モデル: Apache-2.0 (reazon-research)
変換スクリプト: MIT
