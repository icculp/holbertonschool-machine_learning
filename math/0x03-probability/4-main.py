#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('f(1):', e1.pdf(1))

e2 = Exponential(lambtha=2)
print('f(1):', e2.pdf(1))

e3 = Exponential()
print('f(1):', e3.pdf(1))

e4 = Exponential(data=[1, 2], lambtha=2)
print('f(1):', e4.pdf(1))

e5 = Exponential(data)
print('f(1):', e5.pdf(1.5))

e6 = Exponential(data)
print('f(1):', e6.pdf(0))

e7 = Exponential(data)
print('f(1):', e7.pdf(100))
