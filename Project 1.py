from Optimization.Functions import sinusoidfunction as sf
from Optimization.Algorithm import classy, optisolve as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project 1:
# Standard input for project
# [13.8, 8.3, 0.022, 1800, 900, 4.2]

# Data for project
d = np.array(pd.read_csv(r'../Data/project1data.csv'))
# Parameter class for project
para = classy.para(0.0001, 0.19, d, 0, 0, 0, 0)
# Function class to optimize
pr = classy.funct(sf.sinusoid, 'LBFGS', 'strongwolfe', [13.8, 8.3, 0.022, 1800, 900, 4.2], para, 1)

# OPTIMIZE!
a = op.optimize(pr)
x = d[:, 0]
plt.plot(x, sf.sinplot(x, a.input), color = "black")
plt.scatter(x, d[:, 1], color = "red")
plt.show()
