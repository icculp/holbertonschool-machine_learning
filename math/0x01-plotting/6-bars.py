#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

t = np.arange(3)
p1 = plt.bar(t, fruit[0], 0.5, color='red')
p2 = plt.bar(t, fruit[1], 0.5, color='yellow', bottom=fruit[0])
p3 = plt.bar(t, fruit[2], 0.5, color='#ff8000', bottom=fruit[0]+fruit[1])
p4 = plt.bar(t, fruit[3], 0.5, color='#ffe5b4',
             bottom=fruit[0]+fruit[1]+fruit[2])

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(t, ('Farrah', 'Fred', 'Felicia'))
plt.yticks([i for i in range(0, 90, 10)])
plt.legend((p1, p2, p3, p4), ('apples', 'bananas', 'oranges', 'peaches'))
plt.tight_layout()
plt.show()
