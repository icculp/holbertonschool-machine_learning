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

b5 = Binomial(data)
print('F(100):', b5.cdf(100))
print("sp(100):", binom.cdf(100, b5.n, b5.p))
print()

b6 = Binomial(data)
print('F(0):', b6.cdf(0))
print("sp(0):", binom.cdf(0, b6.n, b6.p))
