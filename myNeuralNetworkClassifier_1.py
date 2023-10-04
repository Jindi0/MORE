# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of quantum neural network classifier."""

from __future__ import annotations

from typing import Union, Optional, Callable, Tuple, cast

import numpy as np
import scipy.sparse
from qiskit.algorithms.optimizers import Optimizer
from scipy.sparse import spmatrix
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from qiskit_machine_learning.algorithms.objective_functions import (
    # BinaryObjectiveFunction,
    # OneHotObjectiveFunction,
    # MultiClassObjectiveFunction,
    ObjectiveFunction,
)


from qiskit_machine_learning.algorithms.trainable_model import TrainableModel
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.utils.loss_functions import Loss
from util import resize_vectors

class SparseArray:
    pass

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of quantum neural network classifier."""



from typing import Callable, cast

import numpy as np
import scipy.sparse
from qiskit.algorithms.optimizers import Optimizer, OptimizerResult
from scipy.sparse import spmatrix
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.validation import check_is_fitted




class myNeuralNetworkClassifier(TrainableModel, ClassifierMixin):

    def __init__(
        self,
        neural_network: NeuralNetwork,
        loss: str | Loss = "squared_error",
        one_hot: bool = False,
        optimizer: Optimizer | None = None,
        warm_start: bool = False,
        initial_point: np.ndarray = None,
        callback: Callable[[np.ndarray, float], None] | None = None,
        cluster: bool = True,
        scale: np.ndarray = None
       
    ):
        """
        Args:
            neural_network: An instance of an quantum neural network. If the neural network has a
                one-dimensional output, i.e., `neural_network.output_shape=(1,)`, then it is
                expected to return values in [-1, +1] and it can only be used for binary
                classification. If the output is multi-dimensional, it is assumed that the result
                is a probability distribution, i.e., that the entries are non-negative and sum up
                to one. Then there are two options, either one-hot encoding or not. In case of
                one-hot encoding, each probability vector resulting a neural network is considered
                as one sample and the loss function is applied to the whole vector. Otherwise, each
                entry of the probability vector is considered as an individual sample and the loss
                function is applied to the index and weighted with the corresponding probability.
            loss: A target loss function to be used in training. Default is `squared_error`,
                i.e. L2 loss. Can be given either as a string for 'absolute_error' (i.e. L1 Loss),
                'squared_error', 'cross_entropy', 'cross_entropy_sigmoid', or as a loss function
                implementing the Loss interface.
            one_hot: Determines in the case of a multi-dimensional result of the
                neural_network how to interpret the result. If True it is interpreted as a single
                one-hot-encoded sample (e.g. for 'CrossEntropy' loss function), and if False
                as a set of individual predictions with occurrence probabilities (the index would be
                the prediction and the value the corresponding frequency, e.g. for absolute/squared
                loss). In case of a one-dimensional categorical output, this option determines how
                to encode the target data (i.e. one-hot or integer encoding).
            optimizer: An instance of an optimizer to be used in training. When `None` defaults to SLSQP.
            warm_start: Use weights from previous fit to start next fit.
            initial_point: Initial point for the optimizer to start from.
            callback: a reference to a user's callback function that has two parameters and
                returns ``None``. The callback can access intermediate data during training.
                On each iteration an optimizer invokes the callback and passes current weights
                as an array and a computed value as a float of the objective function being
                optimized. This allows to track how well optimization / training process is going on.
        Raises:
            QiskitMachineLearningError: unknown loss, invalid neural network
        """
        super().__init__(neural_network, loss, optimizer, warm_start, initial_point, callback)
        self._one_hot = one_hot
        
        # encodes the target data if categorical
        self._target_encoder = OneHotEncoder(sparse=False) if one_hot else LabelEncoder()

        # For ensuring the number of classes matches those of the previous
        # batch when training from a warm start.
        self._num_classes: int | None = None

        self._cluster = cluster

        if self._cluster:
            self._loss = myClusterLoss(scale=scale)
        else:
            self._loss = myClassifyLoss()
        

    @property
    def num_classes(self) -> int | None:
        """The number of classes found in the most recent fit.

        If called before :meth:`fit`, this will return ``None``.
        """
        # For user checking and validation.
        return self._num_classes

    # pylint: disable=invalid-name
    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> OptimizerResult:
        X, y = self._validate_input(X, y)

        return self._minimize(X, y)

    def _minimize(self, X: np.ndarray, y: np.ndarray) -> OptimizerResult:
        # mypy definition
        function: ObjectiveFunction = None
        if self._cluster:
            # self._validate_binary_targets(y)
            function = myClusteringObjectiveFunction(X, y, self._neural_network, self._loss)
        else:
            function = mySuperviseObjectiveFunction(X, y, self._neural_network, self._loss)
        
        objective = self._get_objective(function)

        return self._optimizer.minimize(
            fun=objective,
            x0=self._choose_initial_point(),
            jac=function.gradient,
        )

    def predict_vector(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()

        X, _ = self._validate_input(X)

        forward = self._neural_network.forward(X, self._fit_result.x)

        return forward

    
    def evaluation(self, X: np.ndarray, y: np.ndarray, centers):
        pred_vectors = self.predict_vector(X)
        pred_vectors = resize_vectors(pred_vectors)

        classes = list(centers.keys())
        center_points = np.array(list(centers.values()))

        correct = 0

        for i in range(len(pred_vectors)):
            cos_dist = [1 - np.dot(pred_vectors[i], center_points[j].T) for j in range(len(center_points))]
            min_value = min(cos_dist)
            min_index=cos_dist.index(min_value)
            pred_label = classes[min_index]
            if pred_label == str(y[i]):
                correct += 1

        acc = 100 * correct / len(pred_vectors)

        print('Accuracy is {}%'.format(acc))

        return acc





    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        return ClassifierMixin.score(self, X, y, sample_weight)

    def _validate_input(self, X: np.ndarray, y: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Validates and transforms if required features and labels. If arrays are sparse, they are
        converted to dense as the numpy math in the loss/objective functions does not work with
        sparse. If one hot encoding is required, then labels are one hot encoded otherwise label
        are encoded via ``LabelEncoder`` from ``SciKit-Learn``. If labels are strings, they
        converted to numerical representation.

        Args:
            X: features
            y: labels

        Returns:
            A tuple with validated and transformed features and labels.
        """
        if scipy.sparse.issparse(X):
            # our math does not work with sparse arrays
            X = cast(spmatrix, X).toarray()  # cast is required by mypy

        if y is not None:
            if scipy.sparse.issparse(y):
                y = cast(spmatrix, y).toarray()  # cast is required by mypy

            if isinstance(y[0], str):
                y = self._encode_categorical_labels(y)
            elif self._one_hot and not self._validate_one_hot_targets(y, raise_on_failure=False):
                y = self._encode_one_hot_labels(y)

            self._num_classes = self._get_num_classes(y)

        return X, y

    def _encode_one_hot_labels(self, y: np.ndarray):
        # conversion to one hot of the labels is required
        y = y.reshape(-1, 1)
        if self._fit_result is None:
            # the model is being trained, fit first
            self._target_encoder.fit(y)
        y = self._target_encoder.transform(y)

        return y

    def _get_num_classes(self, y: np.ndarray) -> int:
        """Infers the number of classes from the targets.

        Args:
            y: The target values.

        Raises:
            QiskitMachineLearningError: If the number of classes differs from
            the previous batch when using a warm start.

        Returns:
            The number of inferred classes.
        """
        if self._one_hot:
            num_classes = y.shape[-1]
        else:
            num_classes = len(np.unique(y))

        if self._warm_start and self._num_classes is not None and self._num_classes != num_classes:
            raise QiskitMachineLearningError(
                f"The number of classes ({num_classes}) is different to the previous batch "
                f"({self._num_classes})."
            )
        return num_classes





'''
  Objective funciton for binary classification
'''
class myClusteringObjectiveFunction(ObjectiveFunction):
    """An objective function for binary representation of the output,
    e.g. classes of ``-1`` and ``+1``."""

    def __init__(
                self, 
                X: np.ndarray, 
                y: np.ndarray, 
                neural_network: NeuralNetwork, 
                loss: Loss, 
                ) -> None:

        super().__init__(X, y, neural_network, loss)

        # self.scale = scale

    def objective(self, weights: np.ndarray) -> float:
        # predict is of shape (N, 1), where N is a number of samples
        predict_vector = self._neural_network_forward(weights)
        target = np.array(self._y)
        
        return float(np.sum(self._loss(predict_vector, target)) / self._num_samples)



    def gradient(self, weights: np.ndarray) -> np.ndarray:

        # output must be of shape (N, 1), where N is a number of samples
        output = self._neural_network_forward(weights)
        # weight grad is of shape (N, 3, num_weights)
        _, weight_grad = self._neural_network.backward(self._X, weights)

        # loss_gradient is of shape (N, 3)
        loss_gradient = self._loss.gradient(output, self._y.reshape(-1, 1))

        # for the output we compute a dot product(matmul) of loss gradient for this output
        # and weights for this output.
        grad = loss_gradient[:, 0] @ weight_grad[:, 0, :]
        # we keep the shape of (1, num_weights)
        grad = grad.reshape(1, -1) / self._num_samples

        return grad







'''
  Weighted distance between two points on the Bloch Shpere
'''

class myClusterLoss(Loss):
    def __init__(
        self, 
        scale: np.ndarray, 
        ) -> None:

        self._scale = scale

    r"""
    This class computes the L1 loss (i.e. absolute error) for each sample as:

    .. math::

        L = - coeff * Dis(p1, p2)
    """

    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        
        a, b = predict[:, 0, :], predict[:, 1, :]

        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        
        cos_dist = [1 - np.dot(a[i], b[i].T) for i in range(len(a))]
        scale = [self._scale[t[0], t[1]] for t in target]
        loss = -1 * np.multiply(scale, cos_dist) + 2

        return loss

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        # self._validate_shapes(predict, target)
        # 


        return np.sign(predict - target)




'''
  Objective funciton for supervise learning
'''
class mySuperviseObjectiveFunction(ObjectiveFunction):
    """An objective function for binary representation of the output,
    e.g. classes of ``-1`` and ``+1``."""

    def __init__(
                self, 
                X: np.ndarray, 
                y: np.ndarray, 
                neural_network: NeuralNetwork, 
                loss: Loss, 
                ) -> None:

        super().__init__(X, y, neural_network, loss)
        

    def objective(self, weights: np.ndarray) -> float:
        # predict is of shape (N, 1), where N is a number of samples
        predict_vector = self._neural_network_forward(weights)
        target = np.array(self._y)
        
        return float(np.sum(self._loss(predict_vector, target)) / self._num_samples)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        '''
            gradient of supervised learning
        '''

        # output must be of shape (N, 1), where N is a number of samples
        output = self._neural_network_forward(weights)
        # weight grad is of shape (N, 1, num_weights)
        _, weight_grad = self._neural_network.backward(self._X, weights)

        # we reshape _y since the output has the shape (N, 1) and _y has (N,)
        # loss_gradient is of shape (N, 1)
        loss_gradient = self._loss.gradient(output, self._y)

        # for the output we compute a dot product(matmul) of loss gradient for this output
        # and weights for this output. [:, 0]
        # grad = np.multiply(loss_gradient, weight_grad)
        grad = np.array([loss_gradient[i] @ weight_grad[i] for i in range(loss_gradient.shape[0])])
        grad = np.sum(grad, axis=0)
        # grad = loss_gradient @ weight_grad
        # we keep the shape of (1, num_weights)
        grad = grad / self._num_samples

        return grad




class myClassifyLoss(Loss):
    r"""
    This class computes the L1 loss (i.e. absolute error) for each sample as:

    .. math::

        L = Dis(p1, p0)
    """
    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        cos_dist = [1 - np.dot(predict[i], target[i].T) for i in range(len(target))]
        
        

        # a, b = predict[:, 0, :], predict[:, 1, :]

        # a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        # b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        # # cos_dist = 1 - np.dot(a, b.T)
        # cos_dist = [1 - np.dot(a[i], b[i].T) for i in range(len(a))]
        # scale = [self._scale[t[0], t[1]] for t in target]
        # loss = -1 * np.multiply(scale, cos_dist) + 2

        return cos_dist

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        # self._validate_shapes(predict, target)
        # print('call the loss gradient')
        gradients = -1 * np.array(target)

        return gradients