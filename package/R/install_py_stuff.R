# For comments on wsl see infos in my langages/wsl/ subdir
.install_py_stuff <- function(cuda=FALSE, pip=TRUE, test_cuda=cuda) {
  reticulate::install_miniconda() # should be able to control the path?
  reticulate::py_install(packages="scikit-learn", pip=pip) # to import *sklearn*
  reticulate::py_install(packages="matplotlib", pip=pip) # 
  reticulate::py_install(packages="plotnine", pip=pip) # 
  reticulate::py_install(packages="torch", pip=pip) #
  reticulate::py_install(packages="zuko", pip=pip) #
  if (cuda) {
    # https://rdrr.io/cran/aifeducation/src/R/install_and_config.R
    reticulate::conda_install(
      packages = c(
        "pytorch",
        paste0("pytorch-cuda","=","12.1")),
      channel=c("pytorch","nvidia"),
      conda = "auto",
      pip = FALSE)
  }
  if (test_cuda) {
    Itorch <- reticulate::import("torch")
    Itorch$tensor(1, device = "cuda")
  }
}

## To select version in conda_install, use '==':
# reticulate::conda_install(
#   packages = c(
#     "zuko",
#     paste0("zuko","==","1.2.0")),
#   conda = "auto",
#   pip = TRUE)
