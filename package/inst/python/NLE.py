# https://sbi.readthedocs.io/en/latest/api_reference/_autosummary/sbi.inference.NLE_A.html

import os
import torch, numpy as np
from sbi.inference import NLE
from sbi.utils import BoxUniform
import time
import pickle
import sys

device = torch.device("cpu")
inference = NLE(density_estimator="maf", device=device)

def py_to_torch(X, devtype):
    X = np.copy(X) # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
    X = torch.from_numpy(X) # to torch tensor...
    X = X.to(torch.float)
    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    if devtype != "cpu":
        X = X.to(devtype)
        
    return X

def NLE_density_estimation(theta, x, **kwargs):

    theta = py_to_torch(theta, device.type)        
    x = py_to_torch(x, device.type)        

    logLsurf = inference.append_simulations(theta, x).train(
            training_batch_size=128,
            show_train_summary=False
        )

    return logLsurf



def NLE_logL(logLsurf, Y, cond, **kwargs):
    nr = Y.shape[0]
    if (nr == 0):
        return None
    
    cond = py_to_torch(cond, device.type)        
    Y = py_to_torch(Y, device.type)
    logLs = logLsurf.log_prob(Y.unsqueeze(0), cond).detach().numpy()

    return logLs

