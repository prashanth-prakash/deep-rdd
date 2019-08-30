import numpy as np
import multiprocessing as mp

from lib.lif import LIF_Recurrent, ParamsLIF_Recurrent

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
fn_out = './results/spiking_bp_N_%d_batchsize_%d.pkl'%(N, params.batch_size)

for j in range(N):
    print("Iteration: %d"%j)
    train_loss, train_acc = lif.train_BP()
    print("Training loss: %f, training accuracy: %f"%(train_loss, train_acc))

#Save weights and such...
lif.save(fn_out)