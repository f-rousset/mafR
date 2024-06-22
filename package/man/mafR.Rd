\name{mafR}
\alias{mafR}
\alias{mafR-package}

\docType{package}

\title{Interface for masked autoregressive flows}

\description{This wraps Python procedures to train Masked Autoregressive Flows (MAFs, Paramakarios et al. 2017) using the Python package \code{zuko}. 

A suitable Python installation is required. Here is a possible installation procedure:
\preformatted{
  reticulate::install_miniconda() # should be able to control the path?
  reticulate::py_install(packages="scikit-learn", pip=TRUE) # to import *sklearn*
  reticulate::py_install(packages="matplotlib", pip=TRUE) # 
  reticulate::py_install(packages="plotnine", pip=TRUE) # 
  reticulate::py_install(packages="torch", pip=TRUE) #
  reticulate::py_install(packages="zuko", pip=TRUE) #
  
  { # To allow use of cuda:
    reticulate::conda_install(
      packages = c(
        "pytorch",
        paste0("pytorch-cuda","=","12.1")),
      channel=c("pytorch","nvidia"),
      conda = "auto",
      pip = FALSE)
  }

}

}

\references{
Papamakarios, G., D. Sterratt, and I. Murray. 2019. Sequential Neural Likelihood:
Fast Likelihood-free Inference with Autoregressive Flows. Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, PMLR 89:837-848, 2019.
https://doi.org/10.48550/arXiv.1705.07057 ; 
https://proceedings.mlr.press/v89/papamakarios19a.html

Rozet, F., Divo, F., Schnake, S (2023) Zuko: Normalizing flows in PyTorch. https://doi.org/10.5281/zenodo.7625672
}

