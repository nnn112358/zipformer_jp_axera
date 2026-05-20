# 用語集

このプロジェクトに出てくる略語・専門用語の説明。

## ハードウェア

| 用語 | 説明 |
|---|---|
| **AX650N** | AXera-tech の SoC。3 NPU core + 8 ARM A55 CPU + GPU + ISP/VPU 等を統合。 |
| **AX650N PCIe** | AX650N を PCIe カード化した製品。Host PC に挿して使う。本リポジトリの対象。 |
| **NPU** | Neural Processing Unit。ニューラルネット演算専用のアクセラレータ。 |
| **CMM** | Carve-out Memory Module。NPU 専用に確保された連続メモリ領域。 |
| **CPU (本文中)** | AX650N 内蔵の ARM A55(8 core 1.7GHz)を指す。Host PC の x86 ではない。 |
| **AXCLRT** | AX Computing Language Runtime。Host-Board 間の PCIe 通信ライブラリ。 |
| **axcl-smi** | AX 版 nvidia-smi 的なツール。`/usr/bin/axcl/axcl-smi` |

## モデル・フレームワーク

| 用語 | 説明 |
|---|---|
| **Zipformer** | k2-fsa が開発した音声認識用 transformer 変種。downsample-then-upsample 構造で計算量を削減。 |
| **Zipformer2** | Zipformer の改良版。本リポジトリで扱う `reazon-research/japanese-zipformer-base-k2-rs35kh` はこれ。 |
| **CTC** | Connectionist Temporal Classification。フレーム単位の予測を可変長テキストに変換する手法。blank 文字 + 連続削除でデコード。 |
| **RNN-T** | RNN Transducer。CTC と並ぶ ASR 手法。encoder + decoder + joiner の 3 モデル構成。 |
| **icefall** | k2-fsa の ASR レシピ集。Zipformer の原典実装はここ。 |
| **k2** | FST(有限状態トランスデューサ)ライブラリ。CTC/RNN-T のデコードに使う。 |
| **wav2vec2** | Meta(旧 FB)の音声 SSL モデル。本プロジェクトのモデルは wav2vec2 風の feature extractor を持つ。 |

## ONNX / 変換ツール

| 用語 | 説明 |
|---|---|
| **ONNX** | Open Neural Network Exchange。モデルの中間表現フォーマット。 |
| **opset** | ONNX のバージョン番号。op の挙動定義が決まる。今回は 17。 |
| **onnxslim** | ONNX を機械的に最適化するライブラリ(定数畳み込み、不要 op 削除など)。 |
| **onnxsim** | 同種の別ツール。今回は onnxslim を使用。 |
| **onnxruntime (ORT)** | ONNX を CPU/GPU で実行するランタイム。FP32 推論で使用。 |
| **pulsar2** | AXERA-TECH の ONNX → axmodel コンパイラ。Docker image で配布。 |
| **axmodel** | AX 専用モデルフォーマット(`.axmodel`)。ONNX wrapper + .neu mcode + メタデータ。 |
| **mcode** | NPU 専用のマイクロコード命令列。pulsar2 が生成。 |
| **dynamo** | PyTorch 2.x の新しい trace 機構。今回は `dynamo=False`(旧 jit-trace)を使用。 |
| **axengine (pyaxengine)** | Python から axmodel を実行するライブラリ。AXCLRT を呼ぶ薄いラッパ。 |

## 量子化

| 用語 | 説明 |
|---|---|
| **量子化 (quantization)** | FP32 を 整数(U8/U16/S16 等)で表現すること。NPU での演算を高速化する目的。 |
| **U16 / S16 / U8** | unsigned 16bit / signed 16bit / unsigned 8bit。bit 数で精度・サイズが決まる。 |
| **scale / zero_point** | 量子化パラメータ。`fp32 ≈ scale × (int - zero_point)`。 |
| **calibration** | 実データを流して各テンソルの分布を測り、scale/zero_point を決める作業。 |
| **calibration_method** | MinMax / Percentile / MSE / KL の 4 種類。本プロジェクトでは Percentile を採用。 |
| **smooth_quant** | activation の外れ値を weights 側に転嫁して、両方の量子化精度を改善する手法。 |
| **PTQ** | Post-Training Quantization。学習済みモデルを後から量子化する手法。pulsar2 build はこれ。 |
| **QAT** | Quantization-Aware Training。学習中から量子化想定で学習する手法。今回は使わず。 |

## グラフ構造

| 用語 | 説明 |
|---|---|
| **subgraph** | axmodel 内部で連続実行される op の塊。NEU(NPU 実行)と ONNX(CPU 実行)の 2 種類。 |
| **fuse subgraph** | pulsar2 が op を統合して subgraph を作る処理。`fuse 1 subgraph(s)` = NPU で完結。 |
| **NEU** | NPU execution unit。NPU 命令列(.neu)を持つ subgraph。 |
| **Op fusion** | 複数の op を 1 つの NPU 命令に統合すること。Conv + ReLU → 単一 op など。 |
| **dead code elimination (DCE)** | 出力に到達しないノードを削除する最適化。 |
| **constant folding** | 定数だけで計算可能な式を事前計算する最適化。 |

## op の固有名詞

| 用語 | 説明 |
|---|---|
| **AxPow** | pulsar2 の Pow 実装。power=0.25/0.5/2/4 のみ対応(負指数不可)。 |
| **AxQuantizedPow** | U16/S16 量子化された Pow。入力範囲に 0 を含むとエラー。 |
| **AxQuant / AxRequantize / AxDequantize** | 量子化/再量子化/逆量子化 op。U16 ↔ FP32 の橋渡し。 |
| **AxLog / AxExp** | log/exp の AX 実装。NPU 対応だが特定条件で CPU 行き。 |
| **AxGather / GatherElements** | テンソルから index で値を抽出。本モデルでは rel-pos attention で 16 個使用。 |
| **Softplus** | `log(1 + exp(x))`。NPU 直接対応している(本プロジェクトの SwooshL/R 置換先)。 |
| **SwooshL / SwooshR** | Zipformer 固有の活性化関数。`log(1+exp(x-offset)) - 0.08x - bias`。softplus と等価。 |
| **BiasNorm** | Zipformer の LayerNorm 簡素版。学習可能 bias + scale 付き。 |
| **Balancer / Whiten** | Zipformer の学習時 gradient 調整モジュール。推論時は no-op で OK。 |

## 性能指標

| 用語 | 説明 |
|---|---|
| **RTF** | Real Time Factor。`処理時間 / 音声長`。0.01 なら 100× リアルタイム速度。 |
| **TOPS** | Tera Operations Per Second。1兆演算/秒。AX650N の NPU 単体で 18 TOPS @ INT8。 |
| **cos sim** | Cosine similarity。2 つのテンソルの方向類似度。`(a·b)/(|a||b|)`。1.0 で完全一致。 |
| **WER** | Word Error Rate。ASR の精度指標(本プロジェクトは subjective に判定、WER 未計測)。 |
| **mean / median / std** | 統計量。本プロジェクトでは推論時間のバラツキを評価。 |

## その他

| 用語 | 説明 |
|---|---|
| **uv** | Rust 製の Python パッケージマネージャ。pip + venv + pyproject の代替。 |
| **HuggingFace transformers** | NLP/Audio モデルの代表的ライブラリ。`AutoModelForCTC.from_pretrained()` 等。 |
| **trust_remote_code** | HF transformers のオプション。リポジトリ内の Python コードを実行することを許可。 |
| **PAT** | Personal Access Token。GitHub の API 認証用 token。 |
| **fbank** | filter-bank features。古典的音声特徴量。本プロジェクトのモデルは fbank 不要(raw waveform 直入力)。 |
