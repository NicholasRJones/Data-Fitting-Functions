"""""""""
Project 1 Function File:
This file contains the function for computing our objective value and gradient to optimize.
This function is the sum of squares of a sinusoid function fitting a data set.
The parameters of this function are:
    - Offset
    - Amplitude
    - Decay
    - Frequency
    - Chirp
    - Phase
Data is stored using the para class as para.data.
The p input determines which function value you'd like to calculate.
    0 - function value
    1 - gradient
    2 - both
"""""""""

import numpy as np


def sinusoid(x, para, p):
    offset, amp, tau, omega, chirp, phi = x
    time = para.data[:, [0, ]]
    volt = para.data[:, [1, ]]
    theta = (omega + chirp * time) * time + phi
    E = np.exp(-time / tau)
    S = E * np.sin(theta)
    vfit = offset + amp * S
    dv = vfit - volt
    if p == 0:
        return (dv ** 2).sum()
    if p > 0:
        C = E * np.cos(theta)
        g = [0, 0, 0, 0, 0, 0]
        damp = 2 * amp
        g[0] = (2 * dv.sum())
        g[1] = (2 * (dv * S).sum())
        g[2] = ((damp / (tau ** 2)) * (time * dv * S).sum())
        g[3] = (damp * (time * dv * C).sum())
        g[4] = (damp * (time * time * dv * C).sum())
        g[5] = (damp * (dv * C).sum())
        if p > 1:
            f = (dv ** 2).sum()
            return f, g
        return g


def sinplot(x, para):
    f = para[0] + para[1] * np.exp(- x / para[2]) * np.sin((para[3] + para[4] * x) * x + para[5])
    return f
