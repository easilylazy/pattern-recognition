""" Softmax Cross-Entropy Loss Layer """

from warnings import resetwarnings
import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11


def softmax(Input):
    batch_size = Input.shape[0]
    class_num = Input.shape[1]
    data_exp = np.exp(Input)
    sum_data_exp = np.sum(data_exp, axis=1)
    sum_data_exp_reshape = sum_data_exp.reshape(batch_size, 1) * np.ones((1, class_num))
    return data_exp / sum_data_exp_reshape


class SoftmaxCrossEntropyLossLayer:
    def __init__(self):
        self.acc = 0.0
        self.loss = np.zeros(1, dtype="f")

    def forward(self, logit, gt):
        """
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

        ############################################################################
        # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch, and
        # store in self.accu and self.loss respectively.
        # Only return the self.loss, self.accu will be used in solver.py.
        self.logit = logit
        self.gt = gt
        self.res = softmax(logit)
        test_pred = np.argmax(logit, axis=1)
        test_true = np.argmax(gt, axis=1)
        err = test_pred - test_true
        self.acc = err[err == 0].size / logit.shape[0]
        self.loss = -np.sum(np.multiply(gt, np.log(self.res + EPS))) / logit.shape[0]

        ############################################################################

        return self.loss

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        return self.res - self.gt  # np.multiply(self.gt,(self.res-1))

        ############################################################################

