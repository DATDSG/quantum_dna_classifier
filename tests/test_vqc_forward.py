import os
import platform
import numpy as np
import pennylane as qml


def test_vqc_forward_cpu():
    """
    Smoke test: tiny VQC on a CPU-only device returns a normalized probability vector.
    - On Windows we default to 'default.qubit' to avoid native-extension crashes.
    - On other OSes we try 'lightning.qubit' first, then fall back to 'default.qubit'.
    """
    if platform.system() == "Windows":
        dev_name = os.environ.get("PL_TEST_DEVICE", "default.qubit")
    else:
        dev_name = os.environ.get("PL_TEST_DEVICE", "lightning.qubit")

    try:
        dev = qml.device(dev_name, wires=2, shots=None)  # analytic mode for stability
    except Exception:
        dev = qml.device("default.qubit", wires=2, shots=None)

    @qml.qnode(dev, interface="autograd")
    def circ(x):
        qml.RX(x[0], 0)
        qml.RY(x[1], 1)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    x = np.array([0.1, 0.2])
    out1 = circ(x)
    out2 = circ(x)  # run twice to check determinism in analytic mode

    # Shape & normalization
    assert isinstance(out1, (np.ndarray, list))
    out1 = np.asarray(out1, dtype=float)
    assert out1.shape == (4,)
    assert np.isfinite(out1).all()
    assert abs(out1.sum() - 1.0) < 1e-6

    # Determinism (no shots)
    out2 = np.asarray(out2, dtype=float)
    assert np.allclose(out1, out2, atol=1e-9)
