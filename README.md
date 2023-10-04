# MORE
This repo is the code used to produce the results presented in "[MORE: Measurement and Correlation Based
Variational Quantum Circuit for Multi-classification](https://arxiv.org/pdf/2307.11875.pdf)".

## Content
- `MORE_clustering.py` implements the first step of MORE, which involves translating classical labels into quantum labels. It utilizes the variational quantum clustering algorithm to capture interclass correlations.
- `MORE_classification.py` implements the second step of MORE, aimed at enhancing model performance. It performs quantum label-based supervised learning to learn data patterns from the training dataset.
- `myNeuralNetworkClassifier_1.py` and `myNeuralNetworkClassifier_2.py` are tailored QNN classifiers designed for the respective steps 1 and 2 of MORE.
- `myBloch.py` implements the customized Bloch sphere visualization.
- `baseline_binary.py`, `baseline_mul_ancilla.py`, and `baseline_mul_subset.py` construct classifiers serving as baseline methods within this study.
- `model.py` provides the functions of constructing circuits for quantum NN models.
- `data_helper.py` includes the functions for data processing.
- `util.py` contains the functions for recording intermediate results.

