#!/usr/bin/env python3
"""
    Advanced Linear Algebra

    not allowed to im-port any module
    must be done by hand!
"""


def determinant(matrix):
    """ Calculates the determinant of a matrix
        matrix is a square list of lists whose determinant should be calculated
        Returns: the determinant of matrix
    """
    if matrix == [[]]:
        return 1
    if type(matrix) is not list or len(matrix) < 1 or\
            not all(isinstance(x, list) for x in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == len(x) for x in matrix):
        raise ValueError("matrix must be a square matrix")
    copy = list(map(list, matrix))
    dim = len(matrix)
    if dim == 1:
        return matrix[0][0]
    elif dim == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    else:
        for cur in range(dim):
            for i in range(cur + 1, dim):
                if copy[cur][cur] == 0:
                    copy[cur][cur] = 1.0e-10
                curScaler = copy[i][cur] / copy[cur][cur]
                for j in range(dim):
                    copy[i][j] = copy[i][j] - curScaler * copy[cur][j]
        det = 1
        for i in range(dim):
            det *= copy[i][i]
    return round(det)


def cofactor(matrix):
    """ Calculates the matrix of minors
        matrix is a square list of lists whose determinant should be calculated
        Returns: the matrix of minors
    """
    if type(matrix) is not list or len(matrix) < 1 or\
            not all(isinstance(x, list) for x in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == len(x) for x in matrix) or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    copy = list(map(list, matrix))
    dim = len(matrix)
    if dim == 1:
        return [[1]]
    elif dim == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]
    else:
        minors = []
        mins = []
        for i in range(dim):
            for j in range(dim):
                minor = matrix[:i] + matrix[i + 1:]
                for k in range(len(minor)):
                    minor[k] = minor[k][: j] + minor[k][j + 1:]
                minors.append(minor)
            mins.append(minor)
        mm = []
        mmm = []
        for m in range(len(minors)):
            co = 1 if ((m + 1) % 2) else -1
            mm.append(co * determinant(minors[m]))
            if (m + 1) % dim == 0:
                mmm.append(mm)
                mm = []
        return mmm
