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

train_acc = ppn.evaluate(X_train, Y_train)
print('Train set accuracy: %.2f%%' % (train_acc*100))

test_acc = ppn.evaluate(X_test, Y_test)
print('Test set accuracy: %.2f%%' % (test_acc*100))

w, b = ppn.weights, ppn.bias

x0_min = -2
x1_min = ( (-(w[0] * x0_min) - b[0])
          / w[1] )

x0_max = 2
x1_max = ( (-(w[0] * x0_max) - b[0])
          / w[1] )

# x0*w0 + x1*w1 + b = 0
# x1  = (-x0*w0 - b) / w1


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

ax[0].plot([x0_min, x0_max], [x1_min, x1_max])
ax[0].scatter(X_train[Y_train==0, 0], X_train[Y_train==0, 1], label='class 0', marker='o')
ax[0].scatter(X_train[Y_train==1, 0], X_train[Y_train==1, 1], label='class 1', marker='s')

ax[1].plot([x0_min, x0_max], [x1_min, x1_max])
ax[1].scatter(X_test[Y_test==0, 0], X_test[Y_test==0, 1], label='class 0', marker='o')
ax[1].scatter(X_test[Y_test==1, 0], X_test[Y_test==1, 1], label='class 1', marker='s')

ax[1].legend(loc='upper left')
plt.show()
