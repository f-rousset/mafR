\name{mafR}
\alias{mafR}
\alias{mafR-package}

\docType{package}

\title{Interface for masked autoregressive flows}

\description{This wraps Python procedures to train Masked Autoregressive Flows (MAFs, Paramakarios et al. 2017) using the Python package \code{zuko}. It has been tested with version 1.1.0 and 1.2.0 of that package. Note that objects created by its version 1.2.0 cannot be read with its version 1.1.0 (i.e., when saved in and read from \code{pickle} files).

A suitable Python installation is required. Below is a possible installation procedure which may be reasonably portable, as it has worked under Windows, under a linux cluster running on Fedora v.8.8, and under the WSL2-Ubuntu under Windows. On the WSL, additional steps are necessary to make \code{cuda} to work.
% see commented script in 
% D:/home/francois/travail/stats/Infusionplus/mafR/package/R/install_py_stuff.R
% for more details
\preformatted{
reticulate::install_miniconda() # should be able to control the path?
reticulate::py_install(packages="scikit-learn", pip=TRUE) # to import *sklearn*
reticulate::py_install(packages="matplotlib", pip=TRUE) # 
reticulate::py_install(packages="plotnine", pip=TRUE) # 
reticulate::py_install(packages="torch", pip=TRUE) #
reticulate::py_install(packages="zuko", pip=TRUE) #

# To allow use of cuda:
reticulate::conda_install(
    packages = c(
      "pytorch",
      paste0("pytorch-cuda","=","12.1")),
    channel=c("pytorch","nvidia"),
    conda = "auto",
    pip = FALSE)

# To test that cuda can be used:
Itorch <- reticulate::import("torch")
Itorch$tensor(1, device = "cuda")

# To test that libstdc++ is accessible where the following 'module' expects it:
reticulate::import("torch._dynamo")
}
The last import may fail on linux with message
\code{ImportError: cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'}.
If so, seek the libstdc++ library in the r-reticulate architecture (something like
\code{/home/<MY_USER_NAME>/.local/share/r-miniconda/lib/libstdc++.so.6.0.29}),
and either create a symbolic link to it 
as \code{libstdc++.so.6} in (likely path) \code{/usr/lib/x86_64-linux-gnu/},
% sudo ln -sf /home/francois/.local/share/r-miniconda/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
or change the PATH variable:
\code{export LD_LIBRARY_PATH=/home/<MY_USER_NAME>/.local/share/r-miniconda/lib/:$LD_LIBRARY_PATH}.
% export LD_LIBRARY_PATH=/home/francois/.local/share/r-miniconda/lib/:$LD_LIBRARY_PATH
ON the WSL, I found that the symbolic link is overwritten each time the WSL is restarted.  
}

\references{
Papamakarios, G., D. Sterratt, and I. Murray. 2019. Sequential Neural Likelihood:
Fast Likelihood-free Inference with Autoregressive Flows. Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, PMLR 89:837-848, 2019.
https://doi.org/10.48550/arXiv.1705.07057 ; 
https://proceedings.mlr.press/v89/papamakarios19a.html

Rozet, F., Divo, F., Schnake, S (2023) Zuko: Normalizing flows in PyTorch. https://doi.org/10.5281/zenodo.7625672
}

