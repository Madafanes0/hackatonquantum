import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from qiskit_aer import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.visualization import plot_histogram
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt


n = 5
num_qubits = n**2
tsp = Tsp.create_random_instance(n, seed=123)
adj_matrix = nx.to_numpy_array(tsp.graph)
print("distance\n", adj_matrix)

colors = ["r" for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    
def draw_tsp_solution(G, order, colors, pos):
    G2 = nx.DiGraph()
    G2.add_nodes_from(G)
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)

draw_graph(tsp.graph, colors, pos)

plt.show()

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
# Encontrar el resultado más probable (ruta más probable)
most_probable_result = max(counts, key=counts.get)

# Decodificar el resultado en una secuencia de nodos
binary_sequence = most_probable_result[::-1]  # Revertir la cadena si es necesario
node_sequence = [i for i, bit in enumerate(binary_sequence) if bit == '1']

# Agregar el nodo inicial al final para formar un ciclo
node_sequence.append(node_sequence[0])

# Mostrar la secuencia de nodos como la ruta óptima
print("Ruta óptima encontrada:", node_sequence)
