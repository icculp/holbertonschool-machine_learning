#!/usr/bin/env python3

if __name__ == '__main__':
    determinant = __import__('0-determinant').determinant

    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]
    mat7 = [[3, 8], [4, 6]]
    mat8 = [[4, 1], [2, 3]]
    mat9 = [[2, -5, 3], [0, 7, -2], [-1, 4, 1]]
    mat10 = [[0, 0], [0, 0]]
    mat11 = [[1, 1], [1, 1]]
    mat12 = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 6, 4, 8], [3, 1, 1, 2]]

    print(determinant(mat0))
    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat4))
    try:
        determinant(mat5)
    except Exception as e:
        print(e)
    try:
        determinant(mat6)
    except Exception as e:
        print(e)
    print(-14, determinant(mat7))
    print(10, determinant(mat8))
    print(41, determinant(mat9))
    try:
        print(determinant([['a']]))
    except Exception as e:
        print(e)
    print(0, determinant(mat10))
    print(0, determinant(mat11))
    print(72, determinant(mat12))
