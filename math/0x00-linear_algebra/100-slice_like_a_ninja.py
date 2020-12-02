#!/usr/bin/env python3
"""
    Task 15 advanced
"""


def np_slice(matrix, axes={}):
    """ slices matrices """
    m = matrix.copy()
    n = list()
    for k in sorted(axes.keys()):
        val = axes[k]
        if len(val) == 3:
            start = val[0]
            stop = val[1]
            step = val[2]
        elif len(val) == 2:
            start = val[0]
            stop = val[1]
            step = None
        elif len(val) == 1:
            start = None
            stop = val[0]
            step = None
        else:
            start = None
            stop = None
            step = None
        s = slice(start, stop, step)
        '''x = np.moveaxis(m.copy(), k, 0)'''
        slc = [slice(None)] * len(m.shape)
        slc[k] = s
        m = m[tuple(slc)]
        """
        m = array_slice(m, k, start, stop, step)
        """

        '''
        print('mstart')
        print(m)
        print('mend')
        '''

        '''t = np.take(m,indices=i,axis=k)'''
        '''
        n.append(m[tuple(slc)])'''
        '''print(t[s])'''
    '''print(slice(2,))'''
    return m
