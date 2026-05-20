# 量子化と精度

ニューラルネットの重みと中間活性化を **FP32 → U16 整数** に変換する仕組みと、精度を保つ工夫。

## なぜ量子化するか

| | FP32 | **U16** |
|---|---|---|
| ビット数 | 32 | **16** |
| メモリ | 100% | **50%** |
| NPU 演算速度 | 遅い(浮動小数演算ユニット必要) | **速い(整数演算で済む)** |
| 表現精度 | 連続実数 | 65536 段階 |

NPU は整数演算ユニットの方が圧倒的に多いため、量子化することで演算スループットが向上。AX650N の場合 U16 で約 4× 速度向上が期待される。

## 量子化の数式

```
fp32_value ≈ scale × (uint16_value - zero_point)
uint16_value = round(fp32_value / scale) + zero_point
```

ある FP32 テンソルの実測範囲が `[-2.0, +2.0]` なら:

```
scale       = (2.0 - (-2.0)) / 65535 ≈ 6.1e-5
zero_point  = 32768          (= U16 の中央値)

fp32 =  0.0  ↔  uint16 = 32768
fp32 = +1.0  ↔  uint16 = 32768 + 16384 = 49152
fp32 = -1.0  ↔  uint16 = 32768 - 16384 = 16384
```

量子化解像度は `scale ≈ 6.1e-5`。これより細かい変化は表現できない(切り捨て誤差発生)。

## キャリブレーション

各テンソルの動作範囲(`min` / `max`)を実音声で測って scale/zero_point を最適化する作業。

```
                                        実音声での分布
                          ┌─────────────────────────┐
                          │     ██                  │
                          │    ████                 │
                          │   ██████                │
                          │  ████████               │
                          │ ██████████              │
                          │████████████████████████│
                          └─────────────────────────┘
                          -2.0     0     +2.0
                            ↑               ↑
                           min             max
                          を測ってこの範囲を U16 全体に割当
```

### Calibration method の比較

| method | 仕組み | 強み | 弱み |
|---|---|---|---|
| **MinMax** | 全データの min/max | シンプル | 外れ値 1 個で範囲が広がる |
| **Percentile** | 99.99% percentile を使う | **外れ値耐性** | パラメータ tuning |
| **MSE** | 量子化誤差を最小化する scale を grid search | 局所最適 | 遅い |
| **KL** | KL divergence で最適化 | 理論的 | 遅い、収束不安定 |

我々は **Percentile** を採用(test.wav の特殊な分布に強い)。

### Smooth Quant

attention 後の activation は外れ値が出やすい(softmax 後の特定 token が突出する等)。これを直接量子化すると外れ値で scale が拡大し、平均的な値の解像度が落ちる。

`enable_smooth_quant: true` で:
1. activation の外れ値を検出
2. その分のスケーリングを **後段の weights に転嫁** (`α` で重み付け)
3. activation 側は穏やかになり、量子化しやすい分布になる

```
[before] activation = [0.1, 0.1, 50.0, 0.1, 0.1]  → scale 拡大、解像度悪化
[after]  activation = [0.5, 0.5, 1.0, 0.5, 0.5]   → 整った分布
         weights   = (元の) × 適切なスケール
```

## 我々の精度結果

`scripts/05_analyze_precision.py` で `output_u16/quant/debug/precision_analysis_table.txt` を集計:

### 全体分布(1933 レイヤー)

| 指標 | 値 |
|---|---|
| 平均 cos sim | 0.9763 |
| 中央値 | 0.9791 |
| 最小 | 0.6607 |
| 最大 | 1.0000 |
| **最終 logits cos** | **0.9833** |

### 閾値別分布

```
cos sim
1.000 ┤█████████████████████████████ p100
0.99  ┤████████████████████████      p75 (1450 layers)
0.95  ┤██████████████                p25 (1808 layers, 93.5%)
0.90  ┤████                          (1917 layers, 99.2%)
0.70  ┤█                             (1932 layers, 99.9%)
0.66  ┤▏                             (1 layer)
```

### op 種別

| op | 数 | min | median |
|---|---|---|---|
| AxQuant (Linear/Conv 量子化) | 1221 | 0.66 | 0.98 |
| AxReshape | 288 | 0.90 | 0.98 |
| AxTranspose | 211 | 0.90 | 0.98 |
| AxSlice | 190 | 0.87 | 0.99 |
| AxGather (rel-pos) | 16 | 0.99 | **1.00** |
| AxExpand | 6 | 0.91 | 0.97 |

### 解釈

- **大多数の層で cos > 0.95**(健全)
- **負の cos のレイヤーゼロ**(致命的破綻なし)
- 一部の Linear で cos 0.66 まで落ちるが、**下流で誤差が相殺**されて最終 logits は cos 0.98
- 認識結果は FP32 と完全一致

## ハマりどころ:silence padding と Pow

無音(`0.0`)の padding を入れると、LayerNorm の variance 計算で 0 になる:

```
variance = mean((x - mean)^2)   if x is all zero → variance = 0
rsqrt(variance) = 1/√0 = ∞     ← 量子化不能
```

pulsar2 の `AxQuantizedPow` は input 範囲に 0 を含むと `Geometric sequence cannot include zero` でビルド失敗。

**解決**: padding に振幅 0.001 のホワイトノイズを混ぜる:

```python
pad = np.random.standard_normal(T_pad).astype(np.float32) * 0.001
audio_padded = np.concatenate([audio, pad])
```

## ハマりどころ:U16 vs S16

| dtype | 範囲 | 用途 |
|---|---|---|
| **U16** (unsigned) | 0 ~ 65535 | 非負の値(softmax 出力、ReLU 後など) |
| **S16** (signed) | -32768 ~ 32767 | 正負両方の値(中間 activation) |
| **U8** | 0 ~ 255 | 入力(画像 pixel)、軽量モデル |

pulsar2 は op の出力分布に応じて自動で U16/S16 を選ぶ。我々は config では `U16` を default にしてるが、内部で適切に S16 に切り替わってる(`precision_analysis_table.txt` 確認)。

## 補足:`data_type: FP32` を使うべきでない理由

config で `op_type: "Add", data_type: "FP32"` のような指定をすると、そのタイプの op が **すべて CPU subgraph 行き**になる。我々の場合:

- `op_type: "Add"` を FP32 → attention 残差の Add 80+ 個全て CPU 行き
- `op_type: "Softmax"` を FP32 → 16 個全て CPU 行き
- 結果: subgraph 19 個に分裂、AXCLRT runtime ロード不可

**ピンポイントで FP32 にしたい場合は `start_tensor_names` / `end_tensor_names` で範囲指定**するが、これも subgraph 増えるので最小限に。

## 関連ドキュメント

- [PIPELINE.md](PIPELINE.md) — 各ステップの詳細
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — エラー対応
- [BENCHMARK.md](BENCHMARK.md) — 性能評価
