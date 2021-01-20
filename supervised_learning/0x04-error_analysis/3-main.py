#!/usr/bin/env python3

import numpy as np
specificity = __import__('3-specificity').specificity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(specificity(confusion))

    
    lib = np.load('labels_logits.npz')
    labels = np.argmax(lib['labels'], axis=1)
    logits = np.argmax(lib['logits'], axis=1)
    from sklearn.metrics import confusion_matrix
    cnf = confusion_matrix(labels, logits)
    '''print(dir(cnf))
    print(cnf.ravel())'''
    '''def perf_measure(y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
               TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==y_hat[i]==0:
               TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
               FN += 1
        return(TP, FP, TN, FN)
    tp, fp, tn, fn = perf_measure(labels, logits)
    print(tn)
    print(fp)'''
