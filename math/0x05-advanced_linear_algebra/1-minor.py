#!/usr/bin/env python3
"""
    Advanced Linear Algebra

    not allowed to import any module
    must be done by hand!
"""


def determinant(matrix):
    """ Calculates the determinant of a matrix
        matrix is a square list of lists whose determinant should be calculated
        Returns: the determinant of matrix
    """
    ''' copy list beyond axis0'''
    print("determinant", matrix)
    if matrix == [[]]:
        return 1
    if type(matrix) is not list or len(matrix) < 1 or\
            not all(isinstance(x, list) for x in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise TypeError("matrix must be a square matrix")
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
                    copy[cur][cur] = 1.0e-18
                curScaler = copy[i][cur] / copy[cur][cur]
                for j in range(dim):
                    copy[i][j] = copy[i][j] - curScaler * copy[cur][j]
        det = 1.0
        for i in range(dim):
            det *= copy[i][i]
    return int(det)


def minor(matrix):
    """ Calculates the determinant of a matrix
        matrix is a square list of lists whose determinant should be calculated
        Returns: the determinant of matrix
    """
    ''' copy list beyond axis0'''
    if matrix == [[]]:
        return 1
    if type(matrix) is not list or len(matrix) < 1 or\
            not all(isinstance(x, list) for x in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise TypeError("matrix must be a non-empty square matrix")
    copy = list(map(list, matrix))
    dim = len(matrix)
    if dim == 1:
        return [[1]]
    elif dim == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]
    else:
        def minor(array, i, j):
            c = array
            c = c[:i] + c[i+1:]
            for k in range(0,len(c)):
                c[k] = c[k][:j]+c[k][j+1:]
            return c
        minors = []
        for i in range(dim):
            for j in range(dim):
                minor = matrix[:i] + matrix[i + 1:]
                for k in range(len(minor)):
                    minor[k] = minor[k][: j] + minor[k][j + 1:]
                    #print(minor[k])
                    minors.append(minor)
        mm = []
        for m in range(len(minors)):
            mm.append(determinant(minors[m]))
        return mm
        mm = []
        for k in range(0,len(c)):
            print(k)
            c[k] = c[k][:j]+c[k][j+1:]
            print(c[k])
            mm.append(c[k])
            def minor(array,i,j):
                c = array
                c = c[:i] + c[i+1:]
            for k in range(0,len(c)):
                c[k] = c[k][:j]+c[k][j+1:]
            return c
        '''
        for cur in range(dim):
            for i in range(cur + 1, dim):
                if copy[cur][cur] == 0:
                    copy[cur][cur] = 1.0e-18
                curScaler = copy[i][cur] / copy[cur][cur]
                for j in range(dim):
                    copy[i][j] = copy[i][j] - curScaler * copy[cur][j]
        det = 0.0
        for i in range(dim):
            det *= copy[i][i]
        return int(det)
        '''
    return mm
