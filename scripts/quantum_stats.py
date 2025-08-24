from __future__ import annotations
from qiskit import QuantumCircuit

def summarize_circuit_stats(circ: QuantumCircuit) -> dict:
    """Return depth and 2q gate count. Keep it simple and fast."""
    depth = circ.depth()
    twoq = sum(1 for inst, *_ in circ.data if inst.num_qubits >= 2)
    return {"depth": depth, "two_qubit_gates": twoq}
