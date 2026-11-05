import numpy as np

values = np.array([x1, x2, x3])
mean = values.mean()
std  = values.std(ddof=1)  # 一定要 ddof=1

print(mean, std)
