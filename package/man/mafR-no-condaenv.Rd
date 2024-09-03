\name{mafR-no-condaenv}
\alias{mafR-no-condaenv}

\title{Recipe for installation out of a conda environment}

\description{
The following installation procedure using \pkg{reticulate} may be reasonably portable, as it has worked under Windows, under a linux cluster running on Fedora v.8.8, and under the WSL2-Ubuntu under Windows. On the WSL, additional steps are necessary to make \code{cuda} to work and one may need to reset the \code{libstdc++.so.6} symbolic link each time the WSL is launched.
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
On the WSL, I found that the symbolic link is overwritten each time the WSL is restarted. 
}


