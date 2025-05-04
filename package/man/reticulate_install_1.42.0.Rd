\name{reticulate-install-1.42.0}
\alias{reticulate-install-1.42.0}

\title{Recipe for installation out of a conda environment using \code{py_require}}

\description{
The following installation procedure has worked under Windows with \pkg{reticulate} 1.42.2.
% see commented script in 
% D:/home/francois/travail/stats/Infusionplus/mafR/package/R/reticulate_install_1.42.0.R
% for possibly more details.
\preformatted{
reticulate::install_miniconda() 
Sys.setenv(
    "UV_INDEX" = 
      trimws(paste(sep = " ",
                   "https://download.pytorch.org/whl/cu121",
                   Sys.getenv("UV_INDEX")
      )))
reticulate::py_require(packages="scikit-learn") # to import *sklearn*
reticulate::py_require(packages="matplotlib") # 
reticulate::py_require(packages="plotnine") # 
reticulate::py_require(c("tensorflow", "torch", "torchvision", "torchaudio"))
reticulate::py_require(packages="zuko") #
  
# To allow use of cuda:
reticulate::conda_install(
    packages = c(
      "pytorch",
      paste0("pytorch-cuda","=","12.1")), # for GPU requiring cuda 12.1
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


