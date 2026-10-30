import numpy as np


def parity(ref, path, feeds, atol=1e-3):
    # run an onnx file and compare against reference outputs (list of np arrays)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)
    errs = [float(np.abs(a - b).max()) for a, b in zip(ref, got)]
    ok = all(e <= atol for e in errs)
    return ok, errs
