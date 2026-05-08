import numpy as np

# a = np.array([[1, 2, 3], [3, 4, 6]])
# b = np.array([[7], [8], [9]])

# c = np.matmul(a, b)
# print(c)

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)
a_sum1 = np.sum(a, axis=0)
print(a_sum1)

a_sum2 = np.sum(a, axis=1)
print(a_sum2)
