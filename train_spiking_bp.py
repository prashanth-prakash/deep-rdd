import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy.random as rand
import seaborn as sns

from lib.lif import LIF_Recurrent, ParamsLIF_Recurrent
from lib.causal import causaleffect_maxv, causaleffect_maxv_linear, causaleffect_maxv_sp

tau_s = 0.020
dt = 0.001
t = 1
DeltaT = 20

t_filter = np.linspace(0, 1, 2000)
exp_filter = np.exp(-t_filter/tau_s)
exp_filter = exp_filter/np.sum(exp_filter)
ds = exp_filter[0]

params = ParamsLIF_Recurrent(exp_filter, dt = dt)
lif = LIF_Recurrent(params, t = t)

N = 1000 #Number of iterations
eval_every = 3
for j in range(N):
    print("Iteration: %d"%j)
    train_loss, train_acc = lif.train_BP()
    print("Training loss: %f, training accuracy: %f"%(train_loss, train_acc))
