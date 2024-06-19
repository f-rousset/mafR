\name{get_py_MAF_handle}
\alias{get_py_MAF_handle}
\title{
  Utilities to manage Python environment and torch tensors
}
\description{
  Utility initializing a Python environment for running \code{zuko.flows.MAF} and retrieving it.
}
\usage{
  get_py_MAF_handle(envir, reset=FALSE, torch_device="cpu")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{envir}{An environment (in the R sense) initialized as shown in the Examples.}
  \item{reset}{Boolean: Whether to reinitialize the Python session or not.}
  \item{torch_device}{Character: \code{"cpu"}; or a GPU backend, either \code{"cuda"} (or \code{"cuda:0"}, etc.) or \code{"mps"} depending on system capabilities.}
}
\value{
  \code{get_py_MAF_handle} returns the modified input environment. The source code should be examined for details.
}
\examples{
  if (FALSE) { # A suitable Python installation is required.
  
    my_env <- list2env(list(is_set=FALSE),parent = emptyenv())
    # Initialization of Python session:
    my_env <- get_py_MAF_handle(my_env, reset=FALSE, torch_device="cpu")
    # => provides access to:
    my_env$torch # Imported Python package (result of reticulate::import("torch"))
    my_env$device # the torch_device
    # and to internal definitions for MAF training 
    
  }
}
