import argparse
import csv

import numpy as np
from utils.kernels import Sigmoid
from utils.sgd import SGD


def parse_args():
  parser = argparse.ArgumentParser(description='SGD train and inference')
  parser.add_argument('train')
  parser.add_argument('test')
  parser.add_argument('output')
  return parser.parse_args()


def read_dataset(path, is_train):
  dataset = []

  with open(path, 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
      if i == 0:  # Skip header
        continue
      if is_train:
        features = [float(i) for i in row[1:-1]]
        label = float(row[-1])
        dataset.append(features + [label])
      else:
        features = [float(i) for i in row[1:]]
        dataset.append([float(row[0])] + features)

  return dataset


def CrossValidation(dataset, k, scale=False):
  size = dataset.shape[0]
  step = size // k

  for i in range(k):
    idx = np.zeros((size), dtype=np.bool)
    idx[(i * step):((i + 1) * step)] = True
    test = dataset[idx]
    train = dataset[~idx]

    x_train = train[:, :-1]
    y_train = train[:, -1]
    y_train[y_train == -1] = 0

    x_test = test[:, :-1]
    y_test = test[:, -1]
    y_test[y_test == -1] = 0

    if scale:
      mean = x_train.mean(axis=0)
      std = x_train.std(axis=0)
      x_train = (x_train - mean) / std
      x_test = (x_test - mean) / std

    sgd = SGD(x_train, y_train, x_test, y_test)
    sgd.train()


if __name__ == '__main__':
  args = parse_args()

  train_dataset = read_dataset(args.train, is_train=True)
  test_dataset = read_dataset(args.test, is_train=False)

  np.random.seed(666)

  train_dataset = np.array(train_dataset)
  test_dataset = np.array(test_dataset)

  np.random.shuffle(train_dataset)

  x_train = train_dataset[:, :-1]
  y_train = train_dataset[:, -1]
  y_train[y_train == -1] = 0

  x_test = test_dataset[:, 1:]

  sgd = SGD(x_train, y_train, x_train, y_train, batch_size=256, layers=[20], activation_f=[Sigmoid])
  sgd.train(verbose=True, schedule=[[10000, 0.01], [10000, 0.001]])
  prediction = sgd.infer(x_test)
  prediction[prediction == 0] = -1

  with open(args.output, 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"')
    writer.writerow(('ID', 'y'))
    for id, pred in zip(test_dataset[:, 0], prediction):
      writer.writerow((int(id), int(pred)))

