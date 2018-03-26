import numpy as np

from utils.kernels import Sigmoid, ReLU, CrossEntropyLoss


class SGD():
  def __init__(self, x_train, y_train, x_test=None, y_test=None, batch_size=256, layers=[20], activation_f=[Sigmoid]):
    assert len(layers) == len(activation_f)
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.batch_size = batch_size
    self.layers = layers
    self.activation_f = activation_f

    self.weights = []
    self.b = []

  def infer(self, x):
    activations = self._forward(x)

    output = activations[len(activations) - 1]
    output = output.reshape(output.shape[0])
    output[output < 0.5] = 0
    output[output >= 0.5] = 1

    return output

  def _forward(self, x):
    assert len(self.weights) != 0
    activations = [x]

    for i in range(0, len(self.layers) + 1):
      activation = np.matmul(activations[i], self.weights[i]) + self.b[i]
      if i != len(self.layers):
        activation = self.activation_f[i].forward(activation)
      else:
        activation = Sigmoid.forward(activation)
      activations.append(activation)

    return activations

  def train(self, verbose=True, schedule=[[10000, 0.01], [10000, 0.001]]):
    def create_xavier(shape):
      bound = np.sqrt(2. / (shape[0] + shape[1]))
      return np.random.uniform(-bound, bound, size=shape), np.random.uniform(-bound, bound, size=(shape[1],))

    features_size = self.x_train.shape[1]
    self.weights = []
    self.b = []

    dimensions = [features_size]
    dimensions.extend(self.layers)
    dimensions.append(1)

    for i in range(len(dimensions) - 1):
      w, bb = create_xavier(shape=(dimensions[i], dimensions[i + 1]))
      self.weights.append(w)
      self.b.append(bb)

    weights_grads = [None for i in range(len(self.weights))]
    b_grads = [None for i in range(len(self.b))]

    elems = np.arange(self.x_train.shape[0])
    global_step = 0
    for train_steps, step in schedule:
      for iter in range(train_steps):
        # TODO: Batch elements balancing
        selection = np.random.choice(elems, self.batch_size, replace=False)
        x = self.x_train[selection]
        y = self.y_train[selection]

        activations = self._forward(x)

        last = len(activations) - 1
        loss = CrossEntropyLoss.forward(activations[last], y)
        grad = CrossEntropyLoss.grad(activations[last], y)

        if verbose:
          if global_step % 100 == 0:
            if self.x_test is not None and self.y_test is not None:
              prediction = self.infer(self.x_test)
              print('Step = {0}, loss = {1}, accuracy {2}'.format(global_step, loss,
                                                                  np.sum(prediction == self.y_test) / self.y_test.size))
            else:
              print('Step = {0}, loss = {1}'.format(global_step, loss))

        last -= 1
        b_grads[last] = np.mean(grad, 0)
        weights_grads[last] = np.matmul(activations[last].T, grad)
        weights_grads[last] /= self.batch_size

        last -= 1
        while last >= 0:
          grad = self.activation_f[last].grad(activations[last + 1]) * np.matmul(grad, self.weights[last + 1].T)
          b_grads[last] = np.mean(grad, 0)
          weights_grads[last] = np.matmul(activations[last].T, grad)
          weights_grads[last] /= self.batch_size
          last -= 1

        for i in range(len(self.weights)):
          self.weights[i] -= step * weights_grads[i]
          self.b[i] -= step * b_grads[i]

        global_step += 1

    return loss