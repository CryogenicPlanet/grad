import numpy as np

default_grad = np.ones((2, 3))

a = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])


b = np.array([
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0]
])

print("a_t", a.transpose(), default_grad)

a_grad = default_grad @ b.transpose()
b_grad = a.transpose() @ default_grad

print("a_grad", a_grad)
print("b_grad", b_grad)

# c = a @ b
# print(c)


# Transposed matrix (A^T)
A_T = np.array([[1, 4],
                [2, 5],
                [3, 6]])

# 2x3 matrix of 1s (B)
B = np.ones((2, 3), dtype=int)

# Matrix multiplication
result = A_T @ B

print("Transposed matrix (A^T):")
print(A_T)

print("\n2x3 matrix of 1s (B):")
print(B)

print("\nResult of matrix multiplication (A^T × B):")
print(result)
