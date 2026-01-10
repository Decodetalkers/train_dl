import numpy as np

x = [1, 2, 3, 4, 5]
w = [3, 4, 5, 6, 8]

z = sum(x_i * w_i for x_i, w_i in zip(x, w))

print(z)

x_vec, w_vec = np.array(x), np.array(w)

z = (x_vec.transpose()).dot(w_vec)

print(z)

z = x_vec.dot(w_vec)

print(z)

x_transpose = x_vec.transpose()

print(x_transpose, x_vec)
