#!/usr/bin/env python3
"""
    Error Analysis project
"""
import numpy as np


def f1_score(confusion):
    """ calculates the f1 score for each class in confusion matrix
        confusion is ndarray (classes, classes), rows correct coluns predicted
        Returns: ndarray (classes,) containing f1 score for each class
    """
    totals = sum(sum(label) for label in confusion)
    '''print("totals", totals)'''
    classes = len(confusion)
    '''print("confusion sum", sum(sum(confusion)))'''
    positives = np.sum(confusion, axis=0)
    falsepos = [positives[i] - confusion[i][i] for i in range(classes)]
    fn_tp = np.sum(confusion, axis=1)
    falseneg = [fn_tp[i] - confusion[i][i] for i in range(classes)]
    trueneg = [sum(sum(confusion)) - falseneg[i] -
               falsepos[i] - confusion[i][i] for i in range(classes)]
    '''print(trueneg)'''
    '''trueneg = [trueneg[i] - confusion[i][i] for i in range(classes)]'''
    '''tp_tn = sum(confusion[i][i] for i in range(classes))
    trueneg = [tp_tn - confusion[i][i] for i in range(classes)]'''
    '''print("trueneg", trueneg)'''
    '''falsepos = [positives[i] - confusion[i][i] for i in range(classes)]'''
    '''return [confusion[i][i] / positives[i] for i in range(classes)]'''
    return np.array([confusion[i][i] / (confusion[i][i] +
                    ((1 / 2) * (falsepos[i] +
                     falseneg[i]))) for i in range(classes)])
