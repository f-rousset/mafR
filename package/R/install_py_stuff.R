.install_py_stuff <- function(cuda=FALSE) {
  reticulate::install_miniconda() # should be able to control the path?
  reticulate::py_install(packages="scikit-learn", pip=TRUE) # to import *sklearn*
  reticulate::py_install(packages="matplotlib", pip=TRUE) # 
  reticulate::py_install(packages="plotnine", pip=TRUE) # 
  reticulate::py_install(packages="torch", pip=TRUE) #
  reticulate::py_install(packages="zuko", pip=TRUE) #
  if (cuda) {
    reticulate::conda_install(
      packages = c(
        "pytorch",
        paste0("pytorch-cuda","=","12.1")),
      channel=c("pytorch","nvidia"),
      conda = "auto",
      pip = FALSE)
  }
}
