from __future__ import annotations

import numpy as np


def _elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.minimum(x, 0)) - 1))


def _gemm(a, b, c=None, alpha=1.0, beta=1.0, transA=0, transB=0):
    a = a.T if transA else a
    b = b.T if transB else b
    y = alpha * (a @ b)
    return y if c is None else y + beta * c


OPS = {
    "Gemm": lambda xs, at: _gemm(*xs, **at),
    "MatMul": lambda xs, at: xs[0] @ xs[1],
    "Add": lambda xs, at: xs[0] + xs[1],
    "Sub": lambda xs, at: xs[0] - xs[1],
    "Mul": lambda xs, at: xs[0] * xs[1],
    "Div": lambda xs, at: xs[0] / xs[1],
    "Elu": lambda xs, at: _elu(xs[0], at.get("alpha", 1.0)),
    "Relu": lambda xs, at: np.maximum(xs[0], 0),
    "Tanh": lambda xs, at: np.tanh(xs[0]),
    "Identity": lambda xs, at: xs[0],
}


class OnnxMLP:
    def __init__(self, path: str):
        import onnx
        from onnx import numpy_helper

        m = onnx.load(path)
        g = m.graph
        self.consts = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in g.initializer}
        self.nodes = [(n.op_type, list(n.input), n.output[0],
                       {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}) for n in g.node]
        unknown = {op for op, *_ in self.nodes} - set(OPS)
        if unknown:
            raise ValueError(f"unsupported ONNX ops {sorted(unknown)}")
        self.inp, self.out = g.input[0].name, g.output[0].name
        self.in_dim = g.input[0].type.tensor_type.shape.dim[-1].dim_value

    def __call__(self, x) -> np.ndarray:
        v = dict(self.consts)
        v[self.inp] = np.asarray(x, np.float32).reshape(-1, self.in_dim)
        for op, ins, out, at in self.nodes:
            v[out] = OPS[op]([v[i] for i in ins if i], at)
        return v[self.out]
