"""SwooshLForward/SwooshRForward を softplus 化 + Balancer/Whiten Identity 化 + is_tracing
→ Log/Exp/Equal/Where を一掃した clean ONNX を export。
"""
import sys
import torch
import torch.nn.functional as F
import importlib
from unittest.mock import patch
from transformers import AutoModelForCTC

REPO = "reazon-research/japanese-zipformer-base-k2-rs35kh"
T_SAMPLES = 10 * 16000
OUT = "encoder_ctc.onnx"


class ExportWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_values):
        out = self.m(input_values=input_values, return_dict=False)
        return out[0] if isinstance(out, tuple) else out.logits


def swoosh_l_softplus(x):
    """SwooshLForward 等価: softplus(x-4) - 0.08x - 0.035"""
    return F.softplus(x - 4.0) - 0.08 * x - 0.035


def swoosh_r_softplus(x):
    """SwooshRForward 等価: softplus(x-1) - 0.08x - 0.313261687"""
    return F.softplus(x - 1.0) - 0.08 * x - 0.313261687


def main():
    print(f"Loading {REPO} ...")
    model = AutoModelForCTC.from_pretrained(REPO, trust_remote_code=True)
    model.eval()

    # scaling module を取得して、SwooshLForward / SwooshRForward を差し替え
    # transformers の trust_remote_code でロードされた module を探す
    scaling_mod = None
    for name, m in model.named_modules():
        cls_module = m.__class__.__module__
        if "scaling" in cls_module:
            scaling_mod = importlib.import_module(cls_module)
            break
    if scaling_mod is None:
        # fallback: 直接 import
        sys.path.insert(0, "/home/nnn/.cache/huggingface/hub/models--reazon-research--japanese-zipformer-base-k2-rs35kh/snapshots/df19e126d86994fb72a0a3653fcb31ebe49e6081")
        scaling_mod = importlib.import_module("scaling")
    print(f"scaling module: {scaling_mod.__file__}")

    # 関数差し替え
    scaling_mod.SwooshLForward = swoosh_l_softplus
    scaling_mod.SwooshRForward = swoosh_r_softplus

    # ActivationDropoutAndLinear.forward は __module__ 内の SwooshLForward を直参照しているので、
    # その forward の global namespace の関数を差し替える必要がある
    # ActivationDropoutAndLinear のクラスを取得してその module 内の globals を上書き
    adl_cls = None
    for name, m in model.named_modules():
        if m.__class__.__name__ == "ActivationDropoutAndLinear":
            adl_cls = m.__class__
            break
    if adl_cls is not None:
        adl_module = sys.modules[adl_cls.__module__]
        adl_module.SwooshLForward = swoosh_l_softplus
        adl_module.SwooshRForward = swoosh_r_softplus
        print(f"Patched module globals in: {adl_cls.__module__}")

    # Balancer/Whiten を no-op に
    n_b = n_w = 0
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        if cls == "Balancer":
            m.forward = (lambda self, x: x).__get__(m, m.__class__)
            n_b += 1
        elif cls == "Whiten":
            m.forward = (lambda self, x: x).__get__(m, m.__class__)
            n_w += 1
    print(f"Patched: Balancer={n_b} Whiten={n_w}")

    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, T_SAMPLES, dtype=torch.float32)
    with torch.inference_mode():
        out = wrapper(dummy)
    print(f"output: {tuple(out.shape)}")

    with patch("torch.jit.is_tracing", return_value=True):
        print(f"Exporting to {OUT} ...")
        torch.onnx.export(
            wrapper, (dummy,), OUT,
            input_names=["input_values"], output_names=["logits"],
            dynamic_axes=None, opset_version=17, do_constant_folding=True, verbose=False,
            dynamo=False,
        )
    print("Done.")

    import onnx
    from collections import Counter
    m = onnx.load(OUT, load_external_data=False)
    c = Counter(n.op_type for n in m.graph.node)
    print(f"\nTotal nodes: {len(m.graph.node)}")
    print("Key ops:")
    for op in ["Log", "Exp", "Equal", "Where", "Softplus", "GatherElements",
               "Softmax", "Pow", "Erf", "Loop", "If", "ConstantOfShape"]:
        v = c.get(op, 0)
        if v > 0:
            print(f"  {op:<22} {v}")


if __name__ == "__main__":
    main()
