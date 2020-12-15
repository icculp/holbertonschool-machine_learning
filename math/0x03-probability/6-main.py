#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('Mean:', n1.mean, ', Stddev:', n1.stddev)

n2 = Normal(mean=70, stddev=10)
print('Mean:', n2.mean, ', Stddev:', n2.stddev)

n3 = Normal(data, stddev=-10)
print('Mean:', n3.mean, ', Stddev:', n3.stddev)

n4 = Normal(data, mean=70, stddev=10)
print('Mean:', n4.mean, ', Stddev:', n4.stddev)
