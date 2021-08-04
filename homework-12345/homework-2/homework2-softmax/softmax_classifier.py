import numpy as np
from numpy.lib.function_base import gradient


def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here

    ############################################################################
    N = int(input.shape[0])
    D = int(input.shape[1])
    C = int(label.shape[1])
    loss = 0
    grad = np.zeros([C, D])
    prediction = np.zeros([N, 1])
    for i in range(N):
        sublabel = label[i]
        subinput = input[i]
        k = np.where(sublabel == 1)
        theta_x = np.dot(W.transpose(), subinput)
        exps = np.exp(theta_x)
        h = exps / np.sum(exps)
        loss += np.log(h[k])
        grad += (h - sublabel).reshape(C, 1) * subinput.reshape(1, D)
        if np.where(exps == exps.max())[0].shape[0] != 1:
            gradient = grad.transpose() / N
            loss /= -N
            print(str(i) + " error")
            return loss, gradient, prediction
        else:
            prediction[i] = np.where(exps == exps.max())[0]

    loss /= -N
    loss += lamda * np.sum(W * W) / 2
    gradient = grad.transpose() / N + lamda * W

    return loss, gradient, prediction
