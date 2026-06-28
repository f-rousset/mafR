\name{init_py_env}
\alias{init_py_env}

\title{Recipe for installation using \code{py_require}}

\description{
Initialize the required \code{Python} environment with the required modules, using procedures introduced in \pkg{reticulate} 1.42.0.
}
\usage{
init_py_env(
    module="zuko", # default value will surely change in some future version!
    # Three arguments ignored when module != "zuko":
    cuda=FALSE, test_cuda=cuda, test_dynamo=TRUE, 
    #
    force=FALSE, test_torch=test_cuda || test_dynamo,
    verbose=interactive())
}
\arguments{
\item{module}{Either \code{"zuko"} or \code{"NLE"}.}
\item{cuda}{Boolean: Whether to install cuda device (only needed once).}
\item{test_cuda}{Boolean: whether to test that cuda device can be used.}
\item{test_dynamo}{Boolean: whether to test that \code{"torch._dynamo"} module can be imported. In particular, this test whether libstdc++ is accessible where this module expects it. See Details if the test fails}
\item{test_torch}{Boolean: whether to test that \code{"torch"} module can be imported.}
\item{force}{Boolean: Whether to force re-installation of miniconda.}
\item{verbose}{Boolean: Whether to reprot progress of the different steps of the procedure.}
}

\details{
The the attempt to import the \code{"torch._dynamo"} module may fail on linux with message
\code{ImportError: cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'}.
If so, seek the libstdc++ library in the r-reticulate architecture (something like
\code{/home/<MY_USER_NAME>/.local/share/r-miniconda/lib/libstdc++.so.6.0.29}),
and either create a symbolic link to it 
as \code{libstdc++.so.6} in (likely path) \code{/usr/lib/x86_64-linux-gnu/},
% sudo ln -sf /home/francois/.local/share/r-miniconda/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
or change the PATH variable:
\code{export LD_LIBRARY_PATH=/home/<MY_USER_NAME>/.local/share/r-miniconda/lib/:$LD_LIBRARY_PATH}.
% export LD_LIBRARY_PATH=/home/francois/.local/share/r-miniconda/lib/:$LD_LIBRARY_PATH
On the WSL, I found that the symbolic link is overwritten each time the WSL is restarted. 
}
\value{Returns NULL invisibly.}
\seealso{\code{\link{get_py_MAF_handle}}}
\examples{
# See help("get_py_MAF_handle") for basic usage

# The following test may be useful in broader contexts:
\dontrun{ # too slow for CRAN checks
if ( ! reticulate::py_available(initialize = FALSE)) 
  init_py_env(module="zuko",test_cuda=FALSE)
}
}

