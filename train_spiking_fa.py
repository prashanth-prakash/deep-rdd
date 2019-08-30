import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy.random as rand
import seaborn as sns
import multiprocessing as mp

from lib.lif import LIF_Recurrent, ParamsLIF_Recurrent
from lib.causal import causaleffect_maxv, causaleffect_maxv_linear, causaleffect_maxv_sp

import copy_reg
import types

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

tau_s = 0.020
dt = 0.001
t = 1
DeltaT = 20
parallel = True
n_proc = 4

t_filter = np.linspace(0, 1, 2000)
exp_filter = np.exp(-t_filter/tau_s)
exp_filter = exp_filter/np.sum(exp_filter)
ds = exp_filter[0]

params = ParamsLIF_Recurrent(exp_filter, dt = dt)
lif = LIF_Recurrent(params, t = t, parallel = parallel)


if parallel:
	pool = mp.Pool(processes = n_proc)

N = 1000 #Number of iterations
fn_out = './results/spiking_fa_N_%d_batchsize_%d.pkl'%(N, params.batch_size)

for j in range(N):
    print("Iteration: %d"%j)
    train_loss, train_acc = lif.train_FA()
    print("Training loss: %f, training accuracy: %f"%(train_loss, train_acc))

#Save weights and such...
lif.save(fn_out)