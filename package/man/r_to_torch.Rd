\name{.r_to_torch}
\alias{.r_to_torch}
\title{
  Utility to manage torch tensors
}
\description{
  (Currently not used nor exported) utility converting an R object to a torch tensor.
}
\usage{
  .r_to_torch(x, py_handle, device) 
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{An R object suitable for use in \code{reticulate::r_to_py(x)} (this being
  as indefinite as the \code{\link[reticulate]{r_to_py}} documentation in this respect.)}
  \item{py_handle}{The return value of \code{\link{get_py_MAF_handle}}, or possibly more generally an environment with (at least) elements \code{torch} and \code{device} defined as in such a return value.}
  \item{device}{Character: \code{"cpu"}; or a GPU backend, either \code{"cuda"} (or \code{"cuda:0"}, etc.) or \code{"mps"} depending on system capabilities.}
}
\value{
  \code{r_to_torch} returns a 32-bit floating-point \pkg{torch} tensor allocated on the given \code{device}.
}
\examples{
my_env <- list2env(list(is_set=FALSE),parent = emptyenv())
my_env <- get_py_MAF_handle(my_env, reset=FALSE, torch_device="cpu")
}
