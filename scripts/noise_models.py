# Simple Aer noise models for QSVM experiments.
from __future__ import annotations
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

def simple_noise_model(p1: float = 0.001, p2: float = 0.01, ro01: float = 0.01, ro10: float = 0.01) -> NoiseModel:
    nm = NoiseModel()
    if p1 and p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["x","y","z","h","s","sdg","rx","ry","rz","u","u1","u2","u3","sx"])
    if p2 and p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx","cz","swap","iswap"])
    if (ro01 > 0) or (ro10 > 0):
        ro = ReadoutError([[1 - ro01, ro01], [ro10, 1 - ro10]])
        nm.add_all_qubit_readout_error(ro)
    return nm
