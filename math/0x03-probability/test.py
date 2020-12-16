#!/usr/bin/env python3

def check_len(tuple_x=()):
    if tuple_x is not None:
        if len(tuple_x) == 1:
            tup = (tuple_x[0], 0)
            return(tup)
        else:
            tup = tuple_x
            return(tup)
    elif tuple_x is None:
        tup = (0, 0)
        return(tup)

def addUtuple
