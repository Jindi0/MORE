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
        scale: np.ndarray = None,
        centers = None,
        adaptor_thd=-1,
        center_dist_arr: np.ndarray = None,
        term_weight: int = 0, 
    ):
        super().__init__(neural_network, loss, optimizer, warm_start, initial_point, callback)
        self._one_hot = one_hot
        # encodes the target data if categorical
        self._target_encoder = OneHotEncoder(sparse=False) if one_hot else LabelEncoder()

        # For ensuring the number of classes matches those of the previous
        # batch when training from a warm start.
        self._num_classes: int | None = None

        self._cluster = cluster
        self._adaptor_thd = adaptor_thd
        self._term_weight = term_weight

        if self._cluster:
            self._loss = myClusterLoss(scale=scale)
        else:
            if self._adaptor_thd <= 0:
                self._loss = myClassifyLoss(scale=scale, centers=centers)
            else:
                self._loss = myClassifyLoss_adapt(scale=scale, centers=centers, 
                                                thd=self._adaptor_thd, 
                                                center_dist_arr=center_dist_arr,
                                                 w=self._term_weight)




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

    def _encode_categorical_labels(self, y: np.ndarray):
        # string data is assumed to be categorical

        # OneHotEncoder expects data with shape (n_samples, n_features) but
        # LabelEncoder expects shape (n_samples,) so set desired shape
        y = y.reshape(-1, 1) if self._one_hot else y
        if self._fit_result is None:
            # the model is being trained, fit first
            self._target_encoder.fit(y)
        y = self._target_encoder.transform(y)

        return y

    def _encode_one_hot_labels(self, y: np.ndarray):
        # conversion to one hot of the labels is required
        y = y.reshape(-1, 1)
        if self._fit_result is None:
            # the model is being trained, fit first
            self._target_encoder.fit(y)
        y = self._target_encoder.transform(y)

        return y

    def _validate_output(self, y_hat: np.ndarray) -> np.ndarray:
        try:
            check_is_fitted(self._target_encoder)
            return self._target_encoder.inverse_transform(y_hat).squeeze()
        except NotFittedError:
            return y_hat






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
                loss: Loss
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
    def __init__(
                self, 
                scale: np.ndarray, 
                centers=None
                ) -> None:

        self._scale = scale
        self._centers = centers

    r"""
    This class computes the L1 loss (i.e. absolute error) for each sample as:

    .. math::

        L = Dis(p1, p0)
    """
    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        
        # centers_list = np.array(list(self._centers.values()))
        # centers_id = list(self._centers.keys())

        predict = np.asarray(predict) / np.linalg.norm(predict, axis=1, keepdims=True)

        loss_list = []
        for i in range(len(target)):
            same_c = np.array(self._centers[str(target[i])])
            sameclass_dis = 1 - np.dot(predict[i], same_c.T)
            loss_list.append(sameclass_dis)

        return loss_list

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        # self._validate_shapes(predict, target)
        # print('call the loss gradient')
        gradients = -1 * np.array(target)

        return gradients


class myClassifyLoss_adapt(Loss):
    def __init__(
                self, 
                scale: np.ndarray, 
                centers=None,
                thd=0.2,
                center_dist_arr=None,
                w=1
                ) -> None:

        self._scale = scale
        self._centers = centers
        self._thd = thd
        self._center_dist_arr = center_dist_arr
        self._w = w


    r"""
    This class computes the L1 loss (i.e. absolute error) for each sample as:

    .. math::

        L = Dis(p1, p0)
    """
    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        
        centers_list = np.array(list(self._centers.values()))
        centers_id = [int(k) for k in list(self._centers.keys())]

        predict = np.asarray(predict) / np.linalg.norm(predict, axis=1, keepdims=True)

        loss_list = []
        for i in range(len(target)):
            same_c = np.array(self._centers[str(target[i])])
            sameclass_dis = 1 - np.dot(predict[i], same_c.T)

            mis_dist = 0
            
            if sameclass_dis <= self._thd: # if the readout is close to the target, and it is possible to misclassified to near label
                for j in centers_id:
                    if self._center_dist_arr[target[i], j] <= self._thd and \
                            self._center_dist_arr[target[i], j] > 0:  

                        near_label = np.array(self._centers[str(j)])
                        mis_dist += 1 - np.dot(predict[i], near_label.T)

            loss = (1-self._w) * sameclass_dis - (self._w * mis_dist) 

            loss_list.append(loss)

        return loss_list

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

        return 0

