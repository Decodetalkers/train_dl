import numpy as np
import matplotlib.pyplot as plt
from model import Perception
data = np.genfromtxt('perceptron_toydata.txt', delimiter='\t')
X, Y = data[:, :2], data[:, 2]
Y = Y.astype(np.int64)

print('Class label counts:', np.bincount(Y))
print('X.shape:', X.shape)
print('y.shape:', Y.shape)

shuffle_idx = np.arange(Y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X: np.ndarray = X[shuffle_idx]
Y: np.ndarray =  Y[shuffle_idx]

X_train: np.ndarray
X_test: np.ndarray
Y_train: np.ndarray
Y_test: np.ndarray
X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
Y_train, Y_test = Y[shuffle_idx[:70]], Y[shuffle_idx[70:]]

# Normalize (mean zero, unit variance)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

X_train.std(axis = 0)

plt.scatter(X_train[Y_train==0, 0], X_train[Y_train==0, 1], label='class 0', marker='o')
plt.scatter(X_train[Y_train==1, 0], X_train[Y_train==1, 1], label='class 1', marker='s')
plt.title('Training set')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend()
plt.show()

ppn = Perception(num_features = 2)
ppn.train(X_train, Y_train, epochs=5)

print('Model parameters:\n\n')
print('  Weights: %s\n' % ppn.weights)
print('  Bias: %s\n' % ppn.bias)

tran_acc = ppn.evaluate(X_train, Y_train)
print(tran_acc)
