#!/usr/bin/env python3

if __name__ == '__main__':
    cofactor = __import__('2-cofactor').cofactor

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]
    mat7 = [[3, 1, 1], [-1, 2, 1], [1, 1, 1]]
    mat8 = [[2, 3, 1], [0, 5, 6], [1, 1, 2]]
    mat9 = [[1, 2, 3], [0, 4, 5], [1, 0, 6]]

    print(cofactor(mat1))
    print(cofactor(mat2))
    print(cofactor(mat3))
    print(cofactor(mat4))
    try:
        cofactor(mat5)
    except Exception as e:
        print(e)
    try:
        cofactor(mat6)
    except Exception as e:
        print(e)
    print([[1, 2, -3], [0, 2, -2], [-1, -4, 7]], cofactor(mat7))
    print([[4, 6, -5], [-5, 3, 1], [13, -12, 10]], cofactor(mat8))
    print([[24, 5, -4], [-12, 3, 2], [-2, -5, 4]], cofactor(mat9))
