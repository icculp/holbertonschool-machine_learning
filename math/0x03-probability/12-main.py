#!/usr/bin/env python3

from scipy.stats import binom
import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('F(30):', b1.cdf(30))
print("sp(30):", binom.cdf(30, b1.n, b1.p))
print()

b2 = Binomial(n=50, p=0.6)
print('F(30):', b2.cdf(30))
print("sp(30):", binom.cdf(30, b2.n, b2.p))
print()
b3 = Binomial(data)
print('F(40):', b3.cdf(40))
print("sp(40):", binom.cdf(40, b3.n, b3.p))
print()

b4 = Binomial(data)
print('F(50):', b4.cdf(50))
print("sp(50):", binom.cdf(50, b4.n, b4.p))
