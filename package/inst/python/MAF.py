import torch
import torch.utils.data as data
import numpy as np
import zuko

def MAF_density_estimation(y_train, y_test, features, transforms, hidden_features, randperm, activation, max_epochs, batch_size):
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
    trainloader = data.DataLoader(y_train, batch_size=batch_size, shuffle=True)
    if y_train.is_cuda:
        flow = zuko.flows.MAF(features=features, transforms=transforms, hidden_features=hidden_features, 
                          randperm=randperm, activation=activation).cuda()
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
      context, transforms, hidden_features, randperm, activation, max_epochs, batch_size):
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
    trainset = data.TensorDataset(*(y_train,x_train))
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    if y_train.is_cuda:
        flow = zuko.flows.MAF(features=features, context=context, transforms=transforms, \
                          hidden_features=hidden_features, randperm=randperm, activation=activation).cuda()
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
