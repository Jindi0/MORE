from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes



# convolutional kernel 
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rx(params[0], 0)
    # target.ry(params[1], 0)
    # target.rz(params[1], 0)

    target.rx(params[1], 1)
    # target.ry(params[4], 1)
    # target.rz(params[3], 1)

    target.ryy(params[2], 0, 1)
    target.rzz(params[3], 0, 1)

    # target.rx(params[7], 0)
    # target.ry(params[5], 0)
    target.rz(params[4], 0)

    # target.rx(params[10], 1)
    # target.ry(params[7], 1)
    target.rz(params[5], 1)

    return target


'''
    Convolutional layer
''' 
def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    kernel_para_num = 6
    params = ParameterVector(param_prefix, length=num_qubits * kernel_para_num)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + kernel_para_num)]), [q1, q2])
        qc.barrier()
        param_index += kernel_para_num
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + kernel_para_num)]), [q1, q2])
        qc.barrier()
        param_index += kernel_para_num

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc



def pool_circuit(params):
    target = QuantumCircuit(2)
    # target.rz(-np.pi / 2, 1)
    # target.cx(1, 0)
    # target.rz(params[0], 0)
    # target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[0], 1)

    return target

'''
    Pooling layer
''' 
def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 1

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc



'''
    8-qubit QCNN classifier with single readout qubit using Z-measurement 
'''
def build_qcnn_baseline_8(num_qubits=8):
    feature_map = ZFeatureMap(num_qubits)
    # feature_map.decompose().draw("mpl")

    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)  
    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # state = Statevector(circuit)
    # print(state)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # we decompose the circuit for the QNN to avoid additional data copying
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    return qnn


'''
    QNN classifier with 8 data processing qubits and n ancilla readout qubits
'''
def build_qnn_baseline_ancilla(ancilla_num):
    num_qubits = 8
    feature_map = ZFeatureMap(num_qubits)

    ansatz = QuantumCircuit(num_qubits+ancilla_num, name="Ansatz")
    ansatz.compose(feature_map, range(0, num_qubits), inplace=True)
    nn = RealAmplitudes(num_qubits+ancilla_num, reps=3)
    ansatz.compose(nn, range(0, num_qubits+ancilla_num), inplace=True)

    if ancilla_num == 3:
        observables = [SparsePauliOp.from_list([("IIZ" + "I" * 8, 1)]), 
                        SparsePauliOp.from_list([("IZI" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("ZII" + "I" * 8, 1)])]
    elif ancilla_num == 4:
        observables = [SparsePauliOp.from_list([("IIIZ" + "I" * 8, 1)]), 
                        SparsePauliOp.from_list([("IIZI" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IZII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("ZIII" + "I" * 8, 1)])]
    elif ancilla_num == 5:
        observables = [SparsePauliOp.from_list([("IIIIZ" + "I" * 8, 1)]), 
                        SparsePauliOp.from_list([("IIIZI" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIZII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IZIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("ZIIII" + "I" * 8, 1)])]
    elif ancilla_num == 6:
        observables = [SparsePauliOp.from_list([("IIIIIZ" + "I" * 8, 1)]), 
                        SparsePauliOp.from_list([("IIIIZI" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIIZII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIZIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IZIIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("ZIIIII" + "I" * 8, 1)])]

    elif ancilla_num == 7:
        observables = [SparsePauliOp.from_list([("IIIIIIZ" + "I" * 8, 1)]), 
                        SparsePauliOp.from_list([("IIIIIZI" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIIIZII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIIZIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIZIIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IZIIIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("ZIIIIII" + "I" * 8, 1)])]
    elif ancilla_num == 8:
        observables = [SparsePauliOp.from_list([("IIIIIIIZ" + "I" * 8, 1)]), 
                        SparsePauliOp.from_list([("IIIIIIZI" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIIIIZII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIIIZIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIIZIIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IIZIIIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("IZIIIIII" + "I" * 8, 1)]),
                        SparsePauliOp.from_list([("ZIIIIIII" + "I" * 8, 1)])]


    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=nn.parameters,
    )

    return qnn



'''
    QNN classifier with 8 data processing qubits and n readout qubits (n<=4)
'''
def build_qcnn_baseline_subset4(num_qubits, classes):
    
    feature_map = ZFeatureMap(num_qubits)
    # feature_map.decompose().draw("mpl")

    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)  
    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)


    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    if classes == 3:
        observables = [SparsePauliOp.from_list([("IIZ" + "I" * 5, 1)]), 
                        SparsePauliOp.from_list([("IZI" + "I" * 5, 1)]),
                        SparsePauliOp.from_list([("ZII" + "I" * 5, 1)])]
    elif classes == 4:
        observables = [SparsePauliOp.from_list([("IIIZ" + "I" * 4, 1)]), 
                        SparsePauliOp.from_list([("IIZI" + "I" * 4, 1)]),
                        SparsePauliOp.from_list([("IZII" + "I" * 4, 1)]),
                        SparsePauliOp.from_list([("ZIII" + "I" * 4, 1)])]

    
    # we decompose the circuit for the QNN to avoid additional data copying
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    return qnn


'''
    QNN classifier with 8 data processing qubits and n readout qubits (8>=n>4)
'''
def build_qcnn_baseline_subset8(num_qubits, classes):
    
    feature_map = ZFeatureMap(num_qubits)
    # feature_map.decompose().draw("mpl")

    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)  
    
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    if classes == 5:
        observables = [SparsePauliOp.from_list([("IIIIZ" + "I" * 3, 1)]), 
                        SparsePauliOp.from_list([("IIIZI" + "I" * 3, 1)]),
                        SparsePauliOp.from_list([("IIZII" + "I" * 3, 1)]),
                        SparsePauliOp.from_list([("IZIII" + "I" * 3, 1)]),
                        SparsePauliOp.from_list([("ZIIII" + "I" * 3, 1)])]
    elif classes == 6:
        observables = [SparsePauliOp.from_list([("IIIIIZ" + "I" * 2, 1)]), 
                        SparsePauliOp.from_list([("IIIIZI" + "I" * 2, 1)]),
                        SparsePauliOp.from_list([("IIIZII" + "I" * 2, 1)]),
                        SparsePauliOp.from_list([("IIZIII" + "I" * 2, 1)]),
                        SparsePauliOp.from_list([("IZIIII" + "I" * 2, 1)]),
                        SparsePauliOp.from_list([("ZIIIII" + "I" * 2, 1)])]
    elif classes == 7:
        observables = [SparsePauliOp.from_list([("IIIIIIZ" + "I" * 1, 1)]), 
                        SparsePauliOp.from_list([("IIIIIZI" + "I" * 1, 1)]),
                        SparsePauliOp.from_list([("IIIIZII" + "I" * 1, 1)]),
                        SparsePauliOp.from_list([("IIIZIII" + "I" * 1, 1)]),
                        SparsePauliOp.from_list([("IIZIIII" + "I" * 1, 1)]),
                        SparsePauliOp.from_list([("IZIIIII" + "I" * 1, 1)]),
                        SparsePauliOp.from_list([("ZIIIIII" + "I" * 1, 1)])]
    elif classes == 8:
        observables = [SparsePauliOp.from_list([("IIIIIIIZ", 1)]), 
                        SparsePauliOp.from_list([("IIIIIIZI", 1)]),
                        SparsePauliOp.from_list([("IIIIIZII", 1)]),
                        SparsePauliOp.from_list([("IIIIZIII", 1)]),
                        SparsePauliOp.from_list([("IIIZIIII", 1)]),
                        SparsePauliOp.from_list([("IIZIIIII", 1)]),
                        SparsePauliOp.from_list([("IZIIIIII", 1)]),
                        SparsePauliOp.from_list([("ZIIIIIII", 1)])]

    # we decompose the circuit for the QNN to avoid additional data copying
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    return qnn


'''
    MORE: Build QCNN with single readout qubit (three measurements)
'''
def build_qcnn(num_qubits):
    num_qubits = 8
    feature_map = ZFeatureMap(num_qubits)
    # feature_map.decompose().draw("mpl")

    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)  
    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # state = Statevector(circuit)
    # print(state)

    # observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    observables = [SparsePauliOp.from_list([("X" + "I" * 7, 1)]), 
                    SparsePauliOp.from_list([("Y" + "I" * 7, 1)]), 
                    SparsePauliOp.from_list([("Z" + "I" * 7, 1)])]

    # we decompose the circuit for the QNN to avoid additional data copying
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    return qnn



