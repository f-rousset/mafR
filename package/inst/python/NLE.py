# https://sbi.readthedocs.io/en/latest/api_reference/_autosummary/sbi.inference.NLE_A.html

import os
import torch, numpy as np
from sbi.inference import NLE
from sbi.inference import MarginalTrainer
from sbi.utils import BoxUniform
import time
import pickle
import sys

device = torch.device("cpu")

def py_to_torch(X, devtype):
    X = np.copy(X) # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
    X = torch.from_numpy(X) # to torch tensor...
    X = X.to(torch.float)
    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    if devtype != "cpu":
        X = X.to(devtype)
        
    return X

def NLE_conditional_density_estimation(density, theta, x, **kwargs):
  
    if density is None:
        trainer = NLE(density_estimator="maf", device=device)
    else:
        trainer = density['trainer']

    theta = py_to_torch(theta, device.type)
    x = py_to_torch(x, device.type)        

    pdf = trainer.append_simulations(theta, x).train(
            training_batch_size=128,
            show_train_summary=False
        )
    density = {'trainer': trainer, 
               'pdf': pdf}

    return density

def sbi_density_estimation(density, x, **kwargs):
  
    if density is None:
        trainer = MarginalTrainer(density_estimator="nsf")
    else:
        trainer = density['trainer']
        
    x = py_to_torch(x, device.type)        
    pdf = trainer.append_samples(x).train()
    density = {'trainer': trainer, 
               'pdf': pdf}
               
    return density



def MAF_predict_cond(density, Y, cond, **kwargs):
    nr = Y.shape[0]
    if (nr == 0):
        return None
    
    pdf = density['pdf']
    cond = py_to_torch(cond, device.type)        
    Y = py_to_torch(Y, device.type)
    logLs = pdf.log_prob(Y.unsqueeze(0), cond).detach().numpy()

    return logLs
  
def MAF_predict_nocond(density, Y, **kwargs):
    nr = Y.shape[0]DE
    if (nr == 0):
        return None
    
    pdf = density['pdf']
    Y = py_to_torch(Y, device.type)
    logLs = pdf.log_prob(Y).detach().numpy()

    return logLs

