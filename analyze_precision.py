"""precision_analysis_table.txt を集計。各層/op_type 別の cos 統計、最終 logits cos など。"""
import re
import numpy as np
from collections import defaultdict

PATH = "output_u16/quant/debug/precision_analysis_table.txt"

with open(PATH) as f:
    text = f.read()

rows = []
cur = []
for ln in text.split("\n"):
    if ln.startswith("├"):
        if cur:
            rows.append(cur); cur = []
    elif ln.startswith("│"):
        parts = [p.strip() for p in ln.split("│")[1:-1]]
        if len(parts) == 8:
            cur.append(parts)
if cur:
    rows.append(cur)

parsed = []
for row in rows:
    if not row:
        continue
    op_type = row[0][1].rstrip("…").strip()
    qd = row[0][5].strip()
    try:
        cos = float(row[0][6])
        mse = float(row[0][7].replace("…", "").strip())
    except ValueError:
        continue
    parsed.append({"op_type": op_type, "qd": qd, "cos": cos, "mse": mse})

cosines = np.array([p["cos"] for p in parsed])
print(f"Total layers analyzed: {len(parsed)}")
print()
print("=== Overall cos sim distribution ===")
print(f"  mean:    {cosines.mean():.4f}")
print(f"  median:  {np.median(cosines):.4f}")
print(f"  min:     {cosines.min():.4f}")
print(f"  max:     {cosines.max():.4f}")
print(f"  p10:     {np.percentile(cosines, 10):.4f}")
print(f"  p25:     {np.percentile(cosines, 25):.4f}")
print(f"  p75:     {np.percentile(cosines, 75):.4f}")
print(f"  p90:     {np.percentile(cosines, 90):.4f}")
print()
print("=== Layers below thresholds ===")
for th in [0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5]:
    n = (cosines < th).sum()
    print(f"  cos <  {th}: {n:>4d}  ({n/len(cosines)*100:.1f}%)")
print()

# By op type
by_op = defaultdict(list)
for p in parsed:
    by_op[p["op_type"]].append(p["cos"])
print("=== By op type (median ascending) ===")
print(f"  {'op_type':<25} {'count':>6} {'min':>8} {'median':>8} {'max':>8}")
for op, vals in sorted(by_op.items(), key=lambda x: np.median(x[1])):
    a = np.array(vals)
    print(f"  {op:<25} {len(a):>6} {a.min():>8.4f} {np.median(a):>8.4f} {a.max():>8.4f}")

# By quantized dtype
print()
print("=== By quantized dtype ===")
by_qd = defaultdict(list)
for p in parsed:
    by_qd[p["qd"]].append(p["cos"])
for qd, vals in by_qd.items():
    a = np.array(vals)
    print(f"  {qd:<8} count={len(a):>5}  mean={a.mean():.4f}  median={np.median(a):.4f}  min={a.min():.4f}")

# Worst 10 layers
print()
print("=== Worst 10 layers (lowest cos) ===")
worst = sorted(parsed, key=lambda p: p["cos"])[:10]
for p in worst:
    print(f"  cos={p['cos']:.4f}  qd={p['qd']:<5}  op={p['op_type']}")

# Final output
print()
print("=== Final output (last row) ===")
print(f"  cos={parsed[-1]['cos']:.4f}  op={parsed[-1]['op_type']}  qd={parsed[-1]['qd']}")
