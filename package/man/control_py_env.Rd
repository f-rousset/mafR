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
control_py_env(seed = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seed}{
Numeric: passed (as integer value) to \code{torch.random.manual_seed}.
}
}
\value{
Returns NULL invisibly.
}
\examples{
## This requires a Python installation accessible through 'reticulate'.
# control_py_env(seed=0L)
}
