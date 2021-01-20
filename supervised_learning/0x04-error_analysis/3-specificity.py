#!/usr/bin/env python3
"""
    Error Analysis project
"""
import numpy as np


def specificity(confusion):
    """ calculates the specificity for each class in confusion matrix
        confusion is ndarray (classes, classes), rows correct coluns predicted
        Returns: ndarray (classes,) containing specificity for each class
    """
    totals = sum(sum(label) for label in confusion)
    '''print("totals", totals)'''
    classes = len(confusion)
    '''print("confusion sum", sum(sum(confusion)))'''
    positives = np.sum(confusion, axis=0)
    falsepos = [positives[i] - confusion[i][i] for i in range(classes)]
    fn_tp = np.sum(confusion, axis=1)
    falseneg = [fn_tp[i] - confusion[i][i] for i in range(classes)]
    trueneg = [sum(sum(confusion)) -
               falseneg[i] -
               falsepos[i] -
               confusion[i][i] for i in range(classes)]
    '''print(trueneg)'''
    '''trueneg = [trueneg[i] - confusion[i][i] for i in range(classes)]'''
    '''tp_tn = sum(confusion[i][i] for i in range(classes))
    trueneg = [tp_tn - confusion[i][i] for i in range(classes)]'''
    '''print("trueneg", trueneg)'''
    '''falsepos = [positives[i] - confusion[i][i] for i in range(classes)]'''
    '''return [confusion[i][i] / positives[i] for i in range(classes)]'''
    return np.array([trueneg[i] /
                    (trueneg[i] + falsepos[i]) for i in range(classes)])
