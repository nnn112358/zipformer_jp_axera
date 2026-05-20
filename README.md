# zipformer_jp_axera

日本語 ASR モデル `reazon-research/japanese-zipformer-base-k2-rs35kh` を **AX650N NPU** で動かす変換パイプライン。

## 結果

| | ONNXRuntime (FP32 CPU) | **axmodel (U16 NPU)** |
|---|---|---|
| 推論時間 | 316.6 ms | **71.5 ms** (4.4× faster) |
| RTF | 0.0317 | **0.0071** (138× real-time) |
| 認識精度 | — | cos sim **0.9879** vs FP32 |

入力: 10秒固定 raw waveform、出力: CTC logits → CTC decode で日本語テキスト。

## クイックスタート

```bash
git clone https://github.com/nnn112358/zipformer_jp_axera.git
cd zipformer_jp_axera
uv sync
uv add /path/to/axengine-0.1.3-py3-none-any.whl

# axmodel 既にある場合
uv run python transcribe_axmodel.py audio.wav

# ベンチマーク
uv run python benchmark.py
```

## 0 から axmodel を作る

```bash
# パイプライン全 6 ステップ
uv run python scripts/01_export_onnx.py           # PyTorch → ONNX
uv run python -c "import onnxslim; onnxslim.slim('encoder_ctc.onnx', output_model='encoder_ctc_slim.onnx')"
uv run python scripts/02_postprocess_onnx.py      # Pow 書換
uv run python scripts/03_fold_shape_args.py       # shape 定数化
uv run python scripts/04_generate_calib.py        # キャリブレーション
docker run --rm -v "$(pwd):/data" -w /data pulsar2:5.2 pulsar2 build \
  --input encoder_ctc_folded.onnx --config configs/encoder_u16_min.json \
  --output_dir output_u16 --output_name encoder_u16.axmodel \
  --target_hardware AX650 --compiler.check 0
```

詳細は [`doc/PIPELINE.md`](doc/PIPELINE.md) 参照。

## ディレクトリ

```
.
├── transcribe_axmodel.py    🎙️ 実機推論 CLI
├── benchmark.py             ⏱️ ORT vs NPU 比較
├── scripts/                 🛠️ 変換パイプライン (00〜05)
├── configs/                 ⚙️ pulsar2 build 設定
└── doc/                     📚 ドキュメント
    ├── REPORT.md            ← 全体俯瞰(なぜ・何を・どう)
    ├── PIPELINE.md          ← 各ステップの詳細実装
    ├── QUANTIZATION.md      ← 量子化と精度の話
    ├── TROUBLESHOOTING.md   ← 7 つの落とし穴
    ├── BENCHMARK.md         ← 性能評価の詳細
    └── GLOSSARY.md          ← 用語集
```

## 必要環境

- Ubuntu 22.04 + AX650N PCIe ボード(driver V3.6.4)
- Docker + `pulsar2:5.2` image (10.3 GB)
- Python 3.10 + [uv](https://github.com/astral-sh/uv)
- `axengine==0.1.3` wheel(別途取得)

## License

MIT (このリポジトリ) / Apache-2.0 (モデル重み by Reazon Holdings Inc.)
