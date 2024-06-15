
get_py_MAF_handle <- function(envir, reset=FALSE, torch_device="cpu") {
  if (reset || ! envir$is_set) {
    cat("\nInitializing python session... ")
    MAF_density_estimation <- MAF_conditional_density_estimation <- NULL
    # reticulate::source_python(paste0(Infusion::projpath(),"/../MAF-R/MAF.py"))
    infile <- system.file('python', "MAF.py", package='mafR')
    reticulate::source_python(infile)
    envir$MAF_density_estimation <- MAF_density_estimation
    envir$MAF_conditional_density_estimation <- MAF_conditional_density_estimation
    envir$is_set <- TRUE
    ## Python packages to be called from R
    envir$torch <- reticulate::import("torch")
    envir$gc <- reticulate::import("gc")
    #
    envir$device <- envir$torch$device(torch_device) 
    # Handle to the eval environ of main Python module:
    envir$py_main <- reticulate::import_main(convert = FALSE) 
    cat("done.\n")
  }
  envir
}

r_to_torch <- function(x, py_handle, device) {
  x <- reticulate::r_to_py(x) # to numpy.ndarray...
  x <- x$copy() # "he given NumPy array is not writable, and PyTorch does not support non-writable tensors."
  x <- py_handle$torch$from_numpy(x) # to torch tensor...
  if (device != "cpu") x <- x$to(device)
  x <- x$float()
  return(x)
}

control_py_env <- function(seed=NULL) {
  py_handle <- get_py_MAF_handle()
  if( ! is.null(seed)) {
    abyss <- py_handle$torch$random$manual_seed(as.integer(seed))
  }
  invisible(NULL)
}



