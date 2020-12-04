#!/usr/bin/env python3
"""
    Task 17 advanced
"""


def matrix_shape(matrix):
    """ Return shape as list of integers """
    shape = []
    if type(matrix) != list:
        pass
    else:
        shape.append(len(matrix))
        shape += matrix_shape(matrix[0])
    return shape


def recur(m1, m2, axis, ax):
    """ head hurts """
    '''print('inside recur')'''
    if axis != ax:
        '''print('if')'''
        return [recur(m1[i], m2[i], axis, ax + 1) for i in range(len(m1))]
    else:
        '''
        print('else')
        print(type(mat2))
        print(mat2)
        print(axis)
        print(ax)'''
        m1.extend(m2)
        return m1


def cat_matrices(mat1, mat2, axis=0):
    """ concatenates two matrices """
    from copy import deepcopy
    '''print('30')'''
    new = deepcopy(mat1)
    new2 = deepcopy(mat2)
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    '''print('34')'''
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i == axis:
            continue
        if shape1[i] != shape2[i]:
            return None
    r = recur(new, new2, axis, 0)
    ''''print(type(r))
    print(r)'''
    return r


'''
    if len(matrix_shape(new)) == 1:
        print('23')
        return new + new2
    if axis == 0:
        try:
            if len(new[0]) != len(new2[0]):
                return None
        except Exception:
            return None
    elif axis == 1:
        if len(new) != len(new2):
            return None
        for li in range(len(new)):
            new[li].extend(new2[li])
    else:
        return None
    return new
'''
