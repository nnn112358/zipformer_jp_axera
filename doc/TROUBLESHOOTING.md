# Troubleshooting

このパイプラインを構築する過程で実際に遭遇したエラー 7 種類 + 解決策。

---

## 1. ONNX に `Loop` / `If` が大量発生

### 症状
```python
import onnx
from collections import Counter
m = onnx.load("encoder.onnx")
c = Counter(n.op_type for n in m.graph.node)
print(c.get("Loop"), c.get("If"))   # 16, 16
```

### 原因
`torch.onnx.export(..., dynamo=True)` (PyTorch 2.x のデフォルト) では `torch.jit.is_tracing()` が **False** を返す。modeling 側の `if torch.jit.is_tracing(): ... else: ...` の else 側(`as_strided` 等の動的シェイプ操作)が走り、ONNX export 時に Loop ノードに化ける。

### 解決
```python
from unittest.mock import patch
with patch("torch.jit.is_tracing", return_value=True):
    torch.onnx.export(
        model, (dummy,), "out.onnx",
        dynamo=False,   # 旧 jit-trace ベースを使用
        opset_version=17,
    )
```

`scripts/01_export_onnx.py` 参照。

---

## 2. `Log` / `Exp` / `Equal` / `Where` が各 80 個

### 症状
ONNX op カウントが異様に多い:
```
Log: 80   Exp: 96   Equal: 90   Where: 90
```

### 原因
zipformer の `SwooshLForward` / `SwooshRForward` 活性化関数が **手書きの stable softplus**:

```python
def SwooshLForward(x):
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log()              # ← Log + Exp
    log_sum = torch.where(log_sum == float("inf"),      # ← Equal + Where
                          x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035
```

これが ActivationDropoutAndLinear 80 個から呼ばれ、各 op が 80 倍に。

### 解決
数学的に等価な `F.softplus` で置換:

```python
import torch.nn.functional as F
def swoosh_l_softplus(x):
    return F.softplus(x - 4.0) - 0.08 * x - 0.035

# scaling module の関数を上書き
import sys
adl_module = sys.modules[ActivationDropoutAndLinear.__module__]
adl_module.SwooshLForward = swoosh_l_softplus
adl_module.SwooshRForward = swoosh_r_softplus
```

→ Log/Exp/Equal/Where は ~10 個まで激減、代わりに `Softplus` 80 個(NPU 対応)。

---

## 3. `Pow(x, -0.5)` で pulsar2 build が落ちる

### 症状
```
TileFailException: AxPow, AxPow only supports power with 0.25, 0.5, 2 and 4, however got [-0.5]
    op: /encoder_embed/out_norm/Pow_1
```

### 原因
LayerNorm の rsqrt 計算 `x^(-0.5)`。`AxPow` は負指数非対応、`Reciprocal` も量子化非対応。

### 解決
`Pow(x, -0.5)` → `Sqrt(x)` + `Div(1, _)` に書換:

```python
sqrt_node = helper.make_node("Sqrt", inputs=[x_in], outputs=[sqrt_out])
div_node = helper.make_node("Div", inputs=[ONE_CONST, sqrt_out], outputs=[pow_out])
```

`scripts/02_postprocess_onnx.py` 参照。

---

## 4. `Reshape, shapefn failed`

### 症状
```
Exception: op name: /encoder/balancer/Reshape, Reshape, shapefn failed.
input: {'x': Tensor(FP32, shape=(1, 256, 250)), 
        'shape': Tensor(S64, name=/.../Shape_output_0, shape=(3,))}
attrs: {..., '_const_inputs': ['shape']}
```

### 原因
`shape` 引数が `Shape → Gather → Concat` で計算されており、pulsar2 が constant でないと判定して落ちる。

### 解決
ORT で実評価して定数化:

```python
# 1. shape テンソルを graph output に追加
# 2. ORT_DISABLE_ALL で 1 回 run
# 3. 結果値を initializer として埋め込み
# 4. 元の computing chain を dead code として削除
```

`scripts/03_fold_shape_args.py` 参照。

---

## 5. `axclrtEngineLoadFromFile failed` (board worker SIGTERM)

### 症状
```python
sess = axengine.InferenceSession("encoder.axmodel")
# RuntimeError: axclrtEngineLoadFromFile failed.
```

### 原因確認
`/tmp/axcl/axcl_logs.txt` で `Decode api(4) response failed.`、board 側 log で `Catch signal 15` (SIGTERM)。

板側 firmware が axmodel をロードしようとして **死亡**。原因は multi-subgraph axmodel:

```bash
# inspect で subgraph 数を確認
cd /home/nnn/Desktop/pulsar2_build_project
uv run --with onnx --with protobuf --with flatbuffers --with numpy --with ml_dtypes \
    python -m sim_axengine.axmodel.inspect output_u16/encoder.axmodel | grep subgraphs

# 出力例: "subgraphs (19):"  ← 1 以外は危険
```

AX650N PCIe runtime は **regular axmodel は 1 subgraph 前提**(LLM-format は別 path)。

### 解決
- pulsar2 build の最終ログで `fuse 1 subgraph(s)` を確認
- 出ない場合は ONNX に NPU 非対応 op が残ってる
  - Log, Exp, Equal, Where が多すぎる → `scripts/01_export_onnx.py` の patch 適用
  - `data_type: "FP32"` の `op_type` 指定 → 削除して all-U16 に
- config を `configs/encoder_u16_min.json` の最小指定にする

---

## 6. 量子化精度が低い(認識崩壊)

### 症状
axmodel 推論で blank ばかり出力、認識テキストが空 or 「②のテト」みたいに崩壊。

### 原因と対応

**A. calibration data が貧弱**
- test.wav 1 件由来の 24 サンプル → median cos 0.62、最終 0.30 で完全崩壊
- → 多様な実音声 100 件以上で再 calibration

**B. silence padding が分布を歪める**
- LayerNorm の variance 計算に影響
- → padding に振幅 0.001 のホワイトノイズを混ぜる

**C. calibration_method が MinMax**
- 外れ値で scale 拡大、平均的な値の解像度低下
- → `"calibration_method": "Percentile"` に変更

**D. smooth_quant 無効**
- attention 後の外れ値で activation 量子化が崩れる
- → `"enable_smooth_quant": true`

**E. LayerNorm の scale が量子化されている**
- → `"ln_scale_data_type": "FP32"` で温存(subgraph 増えない)

我々の cos 推移:
- 初期: 0.30(全崩壊)
- B+C+D 適用後: 0.63(部分崩壊、blank 多発)
- E 適用後: **0.98**(認識完璧)

---

## 7. 10秒固定入力で先頭の音節が脱落

### 症状
`こんにちはこれは...` を入力したのに `虹はこれは...` と認識される。

### 原因
モデルは attention で全入力フレームを参照。trailing zero(padding 部分)が attention 計算に影響し、頭の音節が消える。

### 解決
**「冒頭に微小ノイズの padding」** を入れる:

```python
def pad_audio(audio, T=160000, rng=np.random.default_rng(0)):
    if len(audio) < T:
        pad = rng.standard_normal(T - len(audio)).astype(np.float32) * 0.001
        audio = np.concatenate([audio, pad])
    return audio[:T]
```

実験結果:
| padding 方式 | 認識結果 |
|---|---|
| zero (`np.zeros`) | こんにちはこれはテキストのテト ❌ |
| **noise 0.001** | **こんにちはこれはテキスト読み上げのテストです** ✓ |
| repeat audio | れはテトです。❌ |
| center (左右 zero pad) | こんにちはこれはテキスト読み上げのテストです。✓ |

---

## バグ報告するときのテンプレ

問題が解決しない場合、以下の情報を集めると issue 解析に役立つ:

```bash
# 環境
uv run python --version
docker images | grep pulsar2
/usr/bin/axcl/axcl-smi
lspci | grep -i axera

# モデル状態
cd /home/nnn/Desktop/pulsar2_build_project
uv run --with onnx --with protobuf --with flatbuffers --with numpy --with ml_dtypes \
    python -m sim_axengine.axmodel.inspect path/to/model.axmodel | head -30

# pulsar2 build log
grep -E "(Error|Exception|fuse)" build.log

# board side log
mkdir -p /tmp/axcl_dump
/usr/bin/axcl/axcl-smi log -t 2 -o /tmp/axcl_dump
tar tzf /tmp/axcl_dump/dev*_log_*.tar.gz
```

## 関連ドキュメント

- [PIPELINE.md](PIPELINE.md) — 各ステップの詳細実装
- [QUANTIZATION.md](QUANTIZATION.md) — 量子化の仕組み
