"""ONNX の shape/indices 引数を ORT で実評価して定数化する。

pulsar2 は Reshape/Expand/Slice/Tile などの shape 引数が initializer であることを
要求する。固定 shape 入力でも Shape/Gather/Concat 経由で計算される shape は
graph 上は dynamic に見えるため、ここで折りたたむ。
"""
import os
import sys
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

SRC = sys.argv[1] if len(sys.argv) > 1 else "encoder_ctc_final.onnx"
DST = sys.argv[2] if len(sys.argv) > 2 else "encoder_ctc_folded.onnx"
INPUT_NAME = "input_values"
INPUT_SHAPE = (1, 160000)

SHAPE_INPUTS = {
    "Reshape":   [1],
    "Expand":    [1],
    "Tile":      [1],
    "Slice":     [1, 2, 3, 4],
    "Pad":       [1],
    "Resize":    [1, 2, 3],
    "ConstantOfShape": [0],
    "OneHot":    [1],
    "TopK":      [1],
}


def collect_shape_arg_tensors(model):
    inits = {init.name for init in model.graph.initializer}
    tensors = set()
    for n in model.graph.node:
        if n.op_type not in SHAPE_INPUTS:
            continue
        for idx in SHAPE_INPUTS[n.op_type]:
            if idx < len(n.input):
                t = n.input[idx]
                if t and t not in inits:
                    tensors.add(t)
    return tensors


def fold_iter(src_path, dst_path):
    model = onnx.load(src_path)
    targets = collect_shape_arg_tensors(model)
    print(f"  candidates: {len(targets)}")
    if not targets:
        if src_path != dst_path:
            import shutil; shutil.copy(src_path, dst_path)
        return 0

    existing_outputs = {o.name for o in model.graph.output}
    for t in targets:
        if t not in existing_outputs:
            vi = onnx.ValueInfoProto()
            vi.name = t
            model.graph.output.append(vi)

    probe = "_probe_fold.onnx"
    with open(probe, "wb") as f:
        f.write(model.SerializeToString())

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(probe, sess_options=opts, providers=["CPUExecutionProvider"])
    dummy = np.random.RandomState(0).randn(*INPUT_SHAPE).astype(np.float32) * 0.01
    targets = list(targets)
    vals = sess.run(targets, {INPUT_NAME: dummy})
    os.remove(probe)

    # reload clean and inject as initializers
    model = onnx.load(src_path)
    for t, v in zip(targets, vals):
        arr = np.asarray(v)
        if arr.dtype != np.int64:
            arr = arr.astype(np.int64)
        init = numpy_helper.from_array(arr, name=t)
        model.graph.initializer.append(init)

    # Dead node removal
    init_names = {i.name for i in model.graph.initializer}
    new_nodes = []
    for n in model.graph.node:
        if all(o in init_names for o in n.output) and n.output:
            continue
        new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    data_file = dst_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    onnx.save(
        model, dst_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_file),
        size_threshold=1024,
    )
    return len(targets)


print(f"Folding shape args in {SRC} ...")
import shutil
# First iter: src -> dst
print("[iter 1]")
n = fold_iter(SRC, DST)
# Subsequent iter: dst -> dst (in place)
for it in range(2, 6):
    if n == 0:
        break
    print(f"[iter {it}]")
    n = fold_iter(DST, DST)

# Cleanup with onnxslim
import onnxslim
print("Running onnxslim ...")
onnxslim.slim(DST, output_model=DST)

# summary
m = onnx.load(DST, load_external_data=False)
from collections import Counter
c = Counter(n.op_type for n in m.graph.node)
print(f"\nFinal nodes: {len(m.graph.node)}")
print("Problem ops:")
for op in ["Loop","If","Range","ConstantOfShape","GatherElements","Where","Pow","Reciprocal","Shape","SequenceEmpty"]:
    if c.get(op,0)>0: print(f"  {op}: {c[op]}")
