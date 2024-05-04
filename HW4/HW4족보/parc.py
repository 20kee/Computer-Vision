import numpy as np
a = np.array([[6,4,2],
              [9,6,3],
              [12,8,4]])

# b = [1]*len(a)
# new_a = np.c_[a, b]
n = np.c_[a[:, 2], a[:, 2]]
a = a[:, [0,1]] / n
print(a)

ab = []
ab.append([1,2],[3,4])
print(ab)
#print(new_a)