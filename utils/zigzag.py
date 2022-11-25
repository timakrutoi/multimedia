def zigzag(x, s=8):
    order = False
    # arr = [i for i in range(8)] + [i for i in range(6, 0, -1)]
    arr = range(s * 2)
    # print(arr)
    res = []
    for g in arr:
        for i in range(g+1):
            if i > s-1 or g-i > s-1:
                continue
            if order:
                res.append((i, g-i))
            else:
                res.append((g-i, i))
            # print(res[-1])
        order = not order

    return res

import numpy as np

s = 8
b = np.zeros((s,s))
a = zigzag(b, s)

for idx, i in enumerate(a):
    # print(idx, i)
    b[i] = idx

print(a)
# print(b)
