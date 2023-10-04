# MORE
This repo is the code used to produce the results presented in "[MORE: Measurement and Correlation Based
Variational Quantum Circuit for Multi-classification]([https://arxiv.org/pdf/2208.07719.pdf](https://arxiv.org/pdf/2307.11875.pdf)https://arxiv.org/pdf/2307.11875.pdf)".

## Content
- `MORE_clustering.py` builds and trains a regular QNN that follows the circuit structure design in [TensorFlow Quantum Tutorials](https://www.tensorflow.org/quantum/tutorials/mnist#14_encode_the_data_as_quantum_circuits), but uses different encoding method (angle encoding).
- `main_sqnn.py` builds and trains a SQNN that consists of four identical-sized quantum feature extractors and a quantum predictor. The method of data partitioning is illustrated in Fig.5(1st panel) of our paper.
- `main_differsize.py` builds and trains a SQNN that consists of three different-sized quantum feature extractors and a quantum predictor. The method of data partitioning is illustrated in Fig.5(3rd panel) of our paper.
- `data_helper.py` includes the functions for data processing.
- `util.py` contains the functions for recording intermediate results and quantum circuits.

