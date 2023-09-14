from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.visualization import plot_histogram
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt

# Definir el grafo del TSP como una matriz de adyacencia
# Esto es solo un ejemplo simplificado, el grafo real sería más grande
adj_matrix = np.array([[0, 1, 2, 3, 5],
                       [1, 0, 4, 5, 5],
                       [2, 4, 0, 6, 5],
                       [3, 5, 6, 0, 5],
                       [3, 5, 6, 0, 5]])

n = len(adj_matrix)  # Número de nodos

print(n)

# Crear un circuito cuántico
qc = QuantumCircuit(n)

# Inicializar el estado superposición uniforme
for qubit in range(n):
    qc.h(qubit)

# Grover's Algorithm para buscar el camino más corto
for _ in range(int(np.pi/4 * np.sqrt(n))):  # Número de iteraciones de Grover
    # Implementar el oráculo que marca la solución correcta
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i, j] == 0:
                qc.cz(i, j)

    # Implementar la difusión
    for qubit in range(n):
        qc.h(qubit)
        qc.x(qubit)
    qc.h(n-1)
    qc.mct(list(range(n-1)), n-1)  # Implementar una puerta multi-control Toffoli
    qc.h(n-1)
    for qubit in range(n):
        qc.x(qubit)
        qc.h(qubit)

# Medir los qubits
qc.measure_all()

# Simular el circuito
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
job = assemble(compiled_circuit, shots=1024)
result = simulator.run(job).result()
counts = result.get_counts()

# Mostrar los resultados
plot_histogram(counts)

plt.show()