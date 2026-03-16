import numpy as np


def srd_constant(xx, args):
    return args * np.ones_like(xx)


def srd_exponential(xx, exp_fit, last_subhalo):
    return (exp_fit[1] * np.exp(exp_fit[0] / xx * exp_fit[2])
            * (xx >= last_subhalo))
