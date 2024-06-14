\name{mafR}
\alias{mafR}
\alias{mafR-package}

\docType{package}

\title{Interface for masked autoregressive flows}

\description{This wraps Python procedures to train Masked Autoregressive Flows (MAFs) using the Python package \code{zuko}. 

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

