\name{control_py_env}
\alias{control_py_env}
\title{
Python controls
}
\description{
Interface to control variables in a Python environment possibly used by Infusion.
Currently the only implemented control is that of the \pkg{torch} random seed.  
}
\usage{
control_py_env(py_handle, seed = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{py_handle}{
    An R environment that provides access to a Python evaluation environment,
    as produced  by \code{\link{get_py_MAF_handle}}
  }
  \item{seed}{
Numeric: passed (as integer value) to \code{torch.random.manual_seed}.
}
}
\value{
Returns NULL invisibly.
}
\examples{
\dontrun{ # This requires a Python installation accessible through 'reticulate'.

my_env <- list2env(list(is_set=FALSE),parent = emptyenv())
## Initialization of Python session:
py_handle <- get_py_MAF_handle(my_env, reset=FALSE, torch_device="cpu")
# 
control_py_env(py_handle, seed=0L)
}
}
