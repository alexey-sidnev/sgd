import numpy as np


class Sigmoid():
  @staticmethod
  def forward(x):
    return 1. / (1. + np.exp(-x))

  @staticmethod
  def grad(x):
    return x * (1. - x)


class ReLU():
  @staticmethod
  def forward(x):
    return np.maximum(x, 0)

  @staticmethod
  def grad(x):
    res = x.copy()
    res[x < 0] = 0
    res[x > 0] = 1
    return res


class CrossEntropyLoss():
  @staticmethod
  def forward(x, y):
    batch_size = x.shape[0]
    sum = 0.
    for i in range(batch_size):
      if y[i] == 0:
        sum += -np.log(1. - x[i])
      else:
        sum += -np.log(x[i])
    return sum / batch_size

  @staticmethod
  def grad(x, y):
    return (x[:, 0] - y).reshape(x.shape)