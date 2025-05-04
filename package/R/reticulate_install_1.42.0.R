# For comments on wsl see infos in my langages/wsl/ subdir
.reticulate_install_1.42.0 <- function(cuda=FALSE, pip=TRUE, test_cuda=cuda,
                              test_dynamo=TRUE) {
  reticulate::install_miniconda() # should be able to control the path?
  # Sys.getenv("UV_INDEX")
  ## See recipe for case 'Windows_torch' in keras3::use_backend() ;
  ## here assuming the GPU requires cuda v12.1 (.../cu126 not tried)
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

  if (test_cuda) {
    Itorch <- reticulate::import("torch")
    Itorch$tensor(1, device = "cuda")
  }
  if (test_dynamo) {
    reticulate::import("torch._dynamo")
    # If ImportError: cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'
    # then try something like 
    # sudo ln -sf /home/francois/.local/share/r-miniconda/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  }
}

# On cluster with limited disk space in home directory: 
# genotoul installs r-reticulate in /home/frousset/.local/
# so we create a symbolic link as suggested in 
#    https://bioinfo.genotoul.fr/index.php/faq/software_faq/ (cf 'overquota')
# mkdir ~/work/.local
# ln -s ~/work/.local   ~/.local
