import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.providers.basic_provider import BasicSimulator

# Create simulator backend
backend = BasicSimulator()

# Energy range (keV)
energies = np.linspace(0.1, 10, 40)
probabilities = []

def quantum_tunneling(E):
    qc = QuantumCircuit(1, 1)

    # Convert energy to rotation angle
    theta = np.sqrt(E) / 3

    qc.ry(theta, 0)
    qc.measure(0, 0)

    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=1000)
    result = job.result()

    counts = result.get_counts()

    # Probability of measuring |1>
    prob = counts.get('1', 0) / 1000
    return prob

# Run simulation
for E in energies:
    p = quantum_tunneling(E)
    probabilities.append(p)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(energies, probabilities)
plt.title("Quantum Simulation of Proton Tunneling")
plt.xlabel("Energy (keV)")
plt.ylabel("Probability")
plt.grid(True)
plt.show()
