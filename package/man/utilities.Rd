\name{get_py_MAF_handle}
\alias{get_py_MAF_handle}
\title{
  Utilities to manage Python environment and torch tensors
}
\description{
  Utility initializing a Python environment for running \code{zuko.flows.MAF} and retrieving it.
}
\usage{
  get_py_MAF_handle(envir, reset=FALSE, torch_device="cpu", GPU_mem=NULL,
                    verbose = TRUE)
}
\arguments{
  \item{envir}{An environment (in the R sense) initialized as shown in the Examples.}
  \item{reset}{Boolean: Whether to reinitialize the Python session or not.}
  \item{torch_device}{Character: \code{"cpu"}; or a GPU backend, either \code{"cuda"} (or \code{"cuda:0"}, etc.) or \code{"mps"} depending on system capabilities.}
  \item{GPU_mem}{For development purposes (effect is complicated). An amount of (dedicated) GPU memory, in bytes.}
  \item{verbose}{Boolean. Whether to print some messages or not.}
}
\value{
  If successful, \code{get_py_MAF_handle} returns the modified input environment. 
  If sourcing the Python code provided by \pkg{mafR} failed (presumably from trying to use an improperly set-up Python environment), the error condition message is returned.
}
\seealso{\code{\link{init_py_env}}, \code{\link{control_py_env}}}
\examples{
\dontrun{ # too slow for CRAN checks
# Initialization of python virtual environment:
init_py_env(test_cuda=FALSE)

# Initialization of python session (defines functions etc. 
#   in python session and provide access to them from R):    
#
my_env <- list2env(list(is_set=FALSE),parent = emptyenv())
py_handle <- get_py_MAF_handle(my_env, reset=FALSE, torch_device="cpu")

# => get_py_MAF_handle() has added objects in myenv:
my_env$torch # Imported Python package (result of reticulate::import("torch"))
my_env$device # the torch_device
# and functions for MAF training, such as myenv$MAF_density_estimation.

# The returned 'py_handle' can be used by oher functions, e.g.:
control_py_env(py_handle, seed=0L)
# alters the seed in the python session.
}
}
