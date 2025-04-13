from numpy import array
from numpy.linalg import det, inv


def new_vector(i_hat: array, j_hat: array, vector: array) -> array:
    basis = array([i_hat, j_hat]).transpose()
    return basis.dot(vector)

print(new_vector(array([2, 0]), array([0, 1.5]), array([1, 2])))
print(new_vector(array([-2, 1]), array([1, -2]), array([1, 2])))

def determinant(i_hat: array, j_hat: array) -> array:
    basis = array([i_hat, j_hat]).transpose()
    return det(basis)

print(determinant(array([1, 0]), array([2, 2])))

print("да можно")

A = array([
    [3, 1, 0],
    [2, 4, 1],
    [3, 1, 8]
])

B = array([
    54,
    12,
    6
])

X = inv(A).dot(B)

print(X)

print(determinant(array([2, 1]), array([6, 3])), "матрица линейно зависима")