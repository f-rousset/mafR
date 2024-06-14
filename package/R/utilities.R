
get_py_MAF_envir <- function(py_envir, reset=FALSE, torch_device="cpu") {
  if (reset || ! py_envir$is_set) {
    cat("\nInitializing python session... ")
    MAF_density_estimation <- MAF_conditional_density_estimation <- NULL
    # reticulate::source_python(paste0(Infusion::projpath(),"/../MAF-R/MAF.py"))
    infile <- system.file('python', "MAF.py", package='mafR')
    reticulate::source_python(infile)
    py_envir$MAF_density_estimation <- MAF_density_estimation
    py_envir$MAF_conditional_density_estimation <- MAF_conditional_density_estimation
    py_envir$is_set <- TRUE
    ## Python packages to be called from R
    py_envir$torch <- reticulate::import("torch")
    #
    py_envir$device <- py_envir$torch$device(torch_device) 
    cat("done.\n")
  }
  py_envir
}

r_to_torch <- function(x, py_MAF_env, device) {
  x <- reticulate::r_to_py(x) # to numpy.ndarray...
  x <- x$copy() # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
  x <- py_MAF_env$torch$from_numpy(x) # to torchtensor...
  if (device != "cpu") x <- x$to(device)
  x <- x$float()
  return(x)
}

control_py_env <- function(seed=NULL) {
  py_env <- get_py_MAF_envir()
  if( ! is.null(seed)) {
    abyss <- py_env$torch$random$manual_seed(as.integer(seed))
  }
  invisible(NULL)
}



