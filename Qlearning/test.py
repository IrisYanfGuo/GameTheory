import numpy as np

a = np.zeros(3)+1/3
a[1] = 1
print(np.argmax(a))