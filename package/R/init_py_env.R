# For comments on wsl see infos in my langages/wsl/ subdir
.reticulate_install_1.42.0 <- function(
    cuda=FALSE, test_cuda=cuda, test_dynamo=TRUE, force=FALSE, 
    test_torch=test_cuda || test_dynamo,
    verbose=interactive(), ...
    ) {
  if (force || 
      ! length(reticulate::miniconda_path())) {
    if (verbose) cat("\n install_miniconda()... \n")
    reticulate::install_miniconda(force=force) # should be able to control the path?
  }
  # Sys.getenv("UV_INDEX")
  ## See recipe for case 'Windows_torch' in keras3::use_backend() ;
  ## here assuming the GPU requires cuda v12.1 (.../cu126 not tried)
  if (verbose) cat("Declare Python requirements... \n")
  
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
  
  if (cuda) {
    if (verbose) cat("install pytorch-cuda... \n")
    reticulate::conda_install(
      packages = c(
        "pytorch",
        paste0("pytorch-cuda","=","12.1")), # for GPU requiring cuda 12.1
      channel=c("pytorch","nvidia"),
      conda = "auto",
      pip = FALSE)
  }

  if (test_torch && verbose) cat("test torch... \n")
  if (test_torch || test_cuda || test_dynamo) Itorch <- reticulate::import("torch")
  if (test_cuda){ 
    if (verbose) cat("test cuda device... \n")
    Itorch$tensor(1, device = "cuda")
  }
  if (test_dynamo) {
    if (verbose) cat("test torch._dynamo... \n")
    reticulate::import("torch._dynamo")
    # If ImportError: cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'
    # then try something like 
    # sudo ln -sf /home/francois/.local/share/r-miniconda/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  }
  invisible(NULL)
}

# On cluster with limited disk space in home directory: 
# genotoul installs r-reticulate in /home/frousset/.local/
# so we create a symbolic link as suggested in 
#    https://bioinfo.genotoul.fr/index.php/faq/software_faq/ (cf 'overquota')
# mkdir ~/work/.local
# ln -s ~/work/.local   ~/.local

init_py_env <- function(
  cuda=FALSE, test_cuda=cuda, test_dynamo=TRUE, force=FALSE,
  test_torch=test_cuda || test_dynamo,
  verbose=interactive()) {
  mc  <- match.call()
  if (utils::packageVersion("reticulate")>="1.42.0") {
    mc[[1]]  <- get(".reticulate_install_1.42.0", asNamespace("mafR"), inherits=FALSE)
    eval(mc,parent.frame()) # called with all the booleans.
  } else {
    stop("'reticulate' version 1.42.0 or higher is needed to run init_py_env().")
  }
}
