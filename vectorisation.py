import math
import time
import numpy as np
import torch

n = 10_000
a = torch.ones(n)
b = torch.ones(n)

# Inefficient
c = torch.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
t_1 = time.time() - t
print(f"{t_1} seconds")

# efficent, i.e. vectorised
t = time.time()
d = a + b
t_2 = time.time() - t
print(f"{t_2} seconds")

# How much faster?
print(f"vectorisation is faster {t_1/t_2} times")

