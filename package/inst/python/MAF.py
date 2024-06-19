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


def MAF_density_estimation(y_train, y_test, features, transforms, hidden_features, \
    randperm, activation, max_epochs, batch_size, device):
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
    patience = 20
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
      context, transforms, hidden_features, randperm, activation, max_epochs, batch_size,
      device):
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
    patience = 20
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

def MAF_predict_cond(density, Y, cond, device):
    """ 
    Density(Y|condition)
    Parameters:
    - density: a MAF 
    - Y: a numpy array (from reticulate::r_to_py)
    - cond: a numpy array
    - devtype: memory device type used by the MAF
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
    
    Y = py_to_torch(Y, device.type)        
    cond = py_to_torch(cond, device.type)        
    pred = density(cond).log_prob(Y).detach().cpu().numpy()
    return pred

def MAF_predict_nocond(density, Y, device):
    """ 
    Density(X)
    Parameters:
    - density: a MAF 
    - Y: a numpy array (from reticulate::r_to_py)
    - devtype: memory device type used by the MAF
    """
    # Y = Y.copy() # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
    # Y = torch.from_numpy(Y) # to torch tensor...
    # Y = Y.float()
    # # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    # if devtype != "cpu":
    #     Y = Y.to(devtype)
        
    Y = py_to_torch(Y, device.type)        

    pred = density().log_prob(Y).detach().cpu().numpy()
    return pred
  
def MAF_simulate_cond(density, nsim_as_tuple, cond, device):
    """ 
    simulate(Y|cond)
    Parameters:
    - density: a MAF 
    - nsim_as_tuple: from reticulate::tuple
    - given: a numpy array
    - devtype: memory device type used by the MAF
    """
    # cond = cond.copy() # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
    # cond = torch.from_numpy(cond) # to torch tensor...
    # cond = cond.float()
    # # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    # if devtype != "cpu":
    #     cond = cond.to(devtype)
        
    cond = py_to_torch(cond, device.type)        
    sim = density(cond).sample(nsim_as_tuple).cpu().numpy()
    return sim


