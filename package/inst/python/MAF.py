# https://zuko.readthedocs.io/en/stable/tutorials/basics.html

import torch
import torch.utils.data as data
import numpy as np
import zuko

def py_to_torch(X, devtype):
    X = np.copy(X) # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
    X = torch.from_numpy(X) # to torch tensor...
    X = X.to(torch.float)
    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    if devtype != "cpu":
        X = X.to(devtype)
        
    return X

def get_gpu_info(devtype):
    if devtype == "cuda":
        longint_tuple = torch.cuda.mem_get_info()
    elif devtype == "mps":
        lontint_tuple = tuple(torch.mps.current_allocated_memory(),
                              torch.mps.driver_allocated_memory())
    return tuple(map(float, longint_tuple)) # bc R only handles "short" integers


def MAF_density_estimation(y_train, y_test, features, transforms, hidden_features, \
    randperm, max_epochs, batch_size, device, activation=torch.nn.Tanh, \
    patience = 20):
    """
    Train a Masked Autoregressive Flow (MAF) model to estimate the density of y

    Parameters:
    - y_train: The training dataset.
    - y_test: The validation dataset.
    - features: The dimension of y.
    - transforms: The number of transformation blocks in the MAF.
    - hidden_features: A tuple defining the hidden layer sizes in the MAF.
    - randperm: Whether to use random permutation.
    - activation: The activation function to use in the MAF.
    - max_epochs: The maximum number of epochs to train for.
    - batch_size: The batch size to use during training.

    Returns:
    - The trained flow model.
    """
    
    y_train = py_to_torch(y_train, device.type)        
    y_test = py_to_torch(y_test, device.type)        
    
    trainloader = data.DataLoader(y_train, batch_size=batch_size, shuffle=True)
    if y_train.is_cuda:
        flow = zuko.flows.MAF(features=features, transforms=transforms, hidden_features=hidden_features, 
                          randperm=randperm, activation=activation).cuda()
    elif y_train.is_mps:
        flow = zuko.flows.MAF(features=features, transforms=transforms, hidden_features=hidden_features, 
                          randperm=randperm, activation=activation).mps()
    else:
        flow = zuko.flows.MAF(features=features, transforms=transforms, hidden_features=hidden_features, 
                          randperm=randperm, activation=activation)
                          
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        losses = []
        for x in trainloader:
            loss = -flow().log_prob(x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach())
        losses = torch.stack(losses)
        # train_loss_mean = losses.mean().item()
        # train_loss_std = losses.std().item()
        # print(f'Epoch {epoch}: Training loss {train_loss_mean} ± {train_loss_std}')
        
        with torch.no_grad():
            test_loss = -flow().log_prob(y_test).mean().item()
            print(f'Epoch {epoch + 1}: Validation loss {test_loss}')
            
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping after {epoch + 1} epochs due to no improvement in validation loss.')
            break
    
    return flow


def MAF_conditional_density_estimation(y_train, x_train, y_test, x_test, features, \
      context, transforms, hidden_features, randperm, max_epochs, batch_size,
      device, activation=torch.nn.Tanh, \
    patience = 20):
    """
    Train a Masked Autoregressive Flow (MAF) model to estimate the conditional density of y given x.

    Parameters:
    - y_train: The training dataset for y.
    - x_train: The training dataset for x.
    - y_test: The validation dataset for y.
    - x_test: The validation dataset for x.
    - features: The dimension of y.
    - context: The dimension of x.
    - transforms: The number of transformation blocks in the MAF.
    - hidden_features: A tuple defining the hidden layer sizes in the MAF.
    - randperm: Whether to use random permutation.
    - activation: The activation function to use in the MAF.
    - max_epochs: The maximum number of epochs to train for.
    - batch_size: The batch size to use during training.

    Returns:
    - The trained flow model.
    """
    
    y_train = py_to_torch(y_train, device.type)        
    y_test = py_to_torch(y_test, device.type)        
    x_train = py_to_torch(x_train, device.type)        
    x_test = py_to_torch(x_test, device.type)        

    trainset = data.TensorDataset(*(y_train,x_train))
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    if y_train.is_cuda:
        flow = zuko.flows.MAF(features=features, context=context, transforms=transforms, \
                          hidden_features=hidden_features, randperm=randperm, activation=activation).cuda()
    elif y_train.is_mps:
        flow = zuko.flows.MAF(features=features, context=context, transforms=transforms, \
                          hidden_features=hidden_features, randperm=randperm, activation=activation).mps()
    else:
        flow = zuko.flows.MAF(features=features, context=context, transforms=transforms, \
                          hidden_features=hidden_features, randperm=randperm, activation=activation)

    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        losses = []
        for y, x in trainloader:
            # print(x.is_cuda)
            # print(y.is_cuda)
            loss = -flow(x).log_prob(y).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach())
        losses = torch.stack(losses)
        # train_loss_mean = losses.mean().item()
        # train_loss_std = losses.std().item()
        # print(f'Epoch {epoch}: Training loss {train_loss_mean} ± {train_loss_std}')
        
        with torch.no_grad():
            test_loss = -flow(x_test).log_prob(y_test).mean().item()
            print(f'Epoch {epoch + 1}: Validation loss {test_loss}')
            
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping after {epoch + 1} epochs due to no improvement in validation loss.')
            break
    
    return flow

def MAF_predict_cond(density, Y, cond, device, batchsize=4000):
    """ 
    Density(Y|condition)
    Parameters:
    - density: a MAF 
    - Y: a numpy array (from reticulate::r_to_py)
    - cond: a numpy array
    - device: memory device used by the MAF
    """
    # Y = Y.copy() # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
    # Y = torch.from_numpy(Y) # to torch tensor...
    # Y = Y.float()
    # cond = cond.copy() # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
    # cond = torch.from_numpy(cond) # to torch tensor...
    # cond = cond.float()
    # # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    # if devtype != "cpu":
    #     Y = Y.to(devtype)
    #     cond = cond.to(devtype)
    nr = Y.shape[0]
    if (nr == 0):
        return None
    
    cond = py_to_torch(cond, device.type)        
    Y = py_to_torch(Y, device.type)        
    
    nfbatch = nr // batchsize
    chk = (nr % batchsize) > 0
    
    if (chk):
        nbatch = nfbatch+1
    else:
        nbatch = nfbatch

    if (nbatch>1):
        pred = torch.tensor([0] * nr).to(torch.float)
        for it in range(nfbatch): # 0 1 2...
            rnge = range(it*batchsize, (it+1) * batchsize)
            pred[rnge] =  density(cond).log_prob(Y[rnge, ]).detach().cpu()
        if chk:
            rnge = range((it+1)*batchsize, nr)
            pred[rnge] =  density(cond).log_prob(Y[rnge, ]).detach().cpu()
        pred = pred.numpy()
    else:
        pred = density(cond).log_prob(Y).detach().cpu().numpy()
        
    return pred

def MAF_predict_nocond(density, Y, device, batchsize=4000):
    """ 
    Density(X)
    Parameters:
    - density: a MAF 
    - Y: a numpy array (from reticulate::r_to_py)
    - device: memory device used by the MAF
    """
    # Y = Y.copy() # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
    # Y = torch.from_numpy(Y) # to torch tensor...
    # Y = Y.float()
    # # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    # if devtype != "cpu":
    #     Y = Y.to(devtype)
    
    nr = Y.shape[0]
    if (nr == 0):
        return None

        
    Y = py_to_torch(Y, device.type) 
    nfbatch = nr // batchsize
    chk = (nr % batchsize) > 0
    
    if (chk):
        nbatch = nfbatch+1
    else:
        nbatch = nfbatch

    if (nbatch>1):
        pred = torch.tensor([0] * nr).to(torch.float)
        for it in range(nfbatch): # 0 1 2...
            rnge = range(it*batchsize, (it+1) * batchsize)
            pred[rnge] =  density().log_prob(Y[rnge, ]).detach().cpu()
        if chk:
            rnge = range((it+1)*batchsize, nr)
            pred[rnge] =  density().log_prob(Y[rnge, ]).detach().cpu()
        pred = pred.numpy()
    else:
        pred = density().log_prob(Y).detach().cpu().numpy()

    return pred
  
def MAF_simulate_cond(density, nsim_as_tuple, cond, device, batchsize=4000):
    """ 
    simulate(Y|cond)
    Parameters:
    - density: a MAF 
    - nsim_as_tuple: from reticulate::tuple
    - cond: a 1-row numpy array;
        density(<  >1-row cond array >).sample((nsim >1,)).cpu() 
      would produce a 3d array (nsim, n_cond, event_size)
      This fn may not yet fully handle this case and the calling R code may prevent it. 
      To get 1 sample for each of several given's: density(< cond array >).sample((1,)).cpu()
    - device: memory device used by the MAF
    """
    # cond = cond.copy() # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
    # cond = torch.from_numpy(cond) # to torch tensor...
    # cond = cond.float()
    # # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    # if devtype != "cpu":
    #     cond = cond.to(devtype)
        
    cond = py_to_torch(cond, device.type)   
    cond = cond[0, ] # drops dim; otherwise sample() will generate a 3D array 
                     # with nsim (or) batchsize samples for each line of cond
    nsim = nsim_as_tuple[0]
    nfbatch = nsim // batchsize
    chk_n = nsim % batchsize
    chk = chk_n > 0
    
    if (chk):
        nbatch = nfbatch+1
    else:
        nbatch = nfbatch

    nc = postdens().event_shape[0] # guessed this by seeking source for 'sample' in zuko and looking around
    if (nbatch>1):
        sim = torch.tensor([0] * (nsim*nc)).to(torch.float)
        sim = sim.view(nsim, nc)
        for it in range(nfbatch): # 0 1 2...
            rnge = range(it*batchsize, (it+1) * batchsize)
            sim[rnge, ] =  density(cond).sample((batchsize,)).cpu()
        if chk:
            rnge = range((it+1)*batchsize, nsim)
            sim[rnge, ] =  density(cond).sample((chk_n,)).cpu()
        sim = sim.numpy()
    else:
        sim = density(cond).sample(nsim_as_tuple).cpu().numpy()
    return sim

def MAF_transform(density, Y, device):
    """ 
    This returns the transformed points.
    This can be used to diagnose the training, 
    as the result should be ~ gaussian(O,I) for train set.
    Parameters:
    - density: a MAF 
    - Y: a numpy array (from reticulate::r_to_py)
    - device: memory device used by the MAF
    """
    nr = Y.shape[0]
    if (nr == 0):
        return None

    Y = py_to_torch(Y, device.type) 
    trsf = density().transform(Y).detach().cpu().numpy()

    return trsf
  

