\name{get_py_MAF_handle}
\alias{get_py_MAF_handle}
\alias{r_to_torch}
\title{
  Utilities to manage Python environment and torch tensors
}
\description{
  Utilities for (1) initializing a Python environment for running \code{zuko.flows.MAF} and retrieving it;
  (2) converting an R object to a torch tensor.
}
\usage{
  get_py_MAF_handle(envir, reset=FALSE, torch_device="cpu")
  r_to_torch(x, py_handle, device) 
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{envir}{An environment (in the R sense) initialized as shown in the Examples.}
  \item{reset}{Boolean: Whether to reinitialize the Python session or not.}
  \item{torch_device, device}{Character: \code{"cpu"} and \code{"cuda"} (or \code{"cuda:0"}, etc.) can be specified.}
  \item{py_handle}{The return value of \code{get_py_MAF_handle}, or possibly more generally an environment with (at least) elements \code{torch} and \code{device} defined as in such a return value.}
  \item{x}{An R object suitable for use in \code{reticulate::r_to_py(x)} (this being
  as indefinite as the \code{\link[reticulate]{r_to_py}} documentation in this respect.)}
}
\value{
  \code{get_py_MAF_handle} returns the modified input environment. The source code shoudl be examined for details.
  \code{r_to_torch} returns a 32-bit floating-point \pkg{torch} tensor allocated on the given \code{device}.
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
    
    r_to_torch(1, my_env, my_env$device)
  }
}
