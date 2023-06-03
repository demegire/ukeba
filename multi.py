import numpy as np
from utils import new_bid

def multi_pareto_frontier_B_b(p, a, m, u, T, q):

    #p, a, m, u are arrays
    B = 0
    bids = []
    for i in range(len(p)):
        B += (T / u[i]) * np.log(m[i] * u[i] / (u[i] * (m[i] - (1 / ((1 + a[i]) * T)) * np.log((((a[i] + 1) * (1 - q[i]) / (1 - p[i])) - a[i] * (1 - q[i])) / (1 + a[i] * q[i])))))
        bids.append(new_bid(p[i], a[i], m[i], u[i], T, q[i]))

    return B, *bids