"""ONNX 後処理: Pow(-0.5) を Div(1, Sqrt(x)) に書き換える。

pulsar2 の AxPow は power=0.25/0.5/2/4 のみサポートで、負の指数は不可。
LayerNorm の rsqrt 部分 (Pow(x, -0.5)) を Sqrt + Div(1, _) に置き換える。
Reciprocal は pulsar2 量子化非対応のため、Div(1, ...) を使う。
"""
import sys
import onnx
import numpy as np
from onnx import helper, numpy_helper
from collections import Counter

SRC = sys.argv[1] if len(sys.argv) > 1 else "encoder_ctc_slim.onnx"
DST = sys.argv[2] if len(sys.argv) > 2 else "encoder_ctc_final.onnx"

print(f"Loading {SRC} ...")
m = onnx.load(SRC)
inits = {init.name: init for init in m.graph.initializer}

# 1.0 定数 (共有)
ONE_NAME = "_const_one_fp32"
m.graph.initializer.append(
    numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=ONE_NAME)
)

new_nodes = []
n_pow_rewritten = 0
for n in m.graph.node:
    if n.op_type == "Pow" and len(n.input) >= 2:
        exp_inp = n.input[1]
        if exp_inp in inits:
            arr = numpy_helper.to_array(inits[exp_inp])
            if arr.size == 1 and float(arr.item()) == -0.5:
                x_in = n.input[0]
                pow_out = n.output[0]
                sqrt_out = f"{n.name}/sqrt_out"
                sqrt_node = helper.make_node(
                    "Sqrt", inputs=[x_in], outputs=[sqrt_out],
                    name=f"{n.name}/sqrt"
                )
                div_node = helper.make_node(
                    "Div", inputs=[ONE_NAME, sqrt_out], outputs=[pow_out],
                    name=f"{n.name}/div"
                )
                new_nodes.extend([sqrt_node, div_node])
                n_pow_rewritten += 1
                continue
    new_nodes.append(n)

del m.graph.node[:]
m.graph.node.extend(new_nodes)
print(f"Rewrote {n_pow_rewritten} Pow(-0.5) -> Sqrt + Div(1, _)")

# external data 形式で保存 (>2GB 対策)
onnx.save(
    m, DST,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=f"{DST}.data",
    size_threshold=1024,
)
print(f"Saved {DST}")

# summary
m2 = onnx.load(DST, load_external_data=False)
c = Counter(n.op_type for n in m2.graph.node)
print(f"\nTotal nodes: {len(m2.graph.node)}")
print("Remaining problem ops:")
for op in ["Loop", "If", "Range", "ConstantOfShape", "GatherElements", "Where", "Pow", "Reciprocal", "SequenceEmpty"]:
    if c.get(op, 0) > 0:
        print(f"  {op}: {c[op]}")
