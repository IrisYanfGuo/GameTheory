import numpy as np
import copy

import matplotlib.pyplot as plt

t = range(10)
y1 = np.arange(10)
y2 = np.arange(10)*2

plt.figure()
plt.plot(t,y1)
plt.plot(t,y2)
plt.legend(("y1","y2"))

plt.show()
