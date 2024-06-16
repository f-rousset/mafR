
get_py_MAF_handle <- function(envir, reset=FALSE, torch_device="cpu") {
  if (reset || ! envir$is_set) {
    cat("\nInitializing python session... ")
    MAF_density_estimation <- MAF_conditional_density_estimation <- 
      MAF_predict_cond <- MAF_predict_nocond <- MAF_simulate_cond <- py_to_torch <- NULL
    # reticulate::source_python(paste0(Infusion::projpath(),"/../MAF-R/MAF.py"))
    infile <- system.file('python', "MAF.py", package='mafR')
    reticulate::source_python(infile)
    envir$MAF_density_estimation <- MAF_density_estimation
    envir$MAF_conditional_density_estimation <- MAF_conditional_density_estimation
    envir$MAF_predict_cond <- MAF_predict_cond
    envir$MAF_predict_nocond <- MAF_predict_nocond
    envir$MAF_simulate_cond <- MAF_simulate_cond
    envir$py_to_torch <- py_to_torch
    envir$is_set <- TRUE
    ## Python packages to be called from R
    envir$torch <- reticulate::import("torch")
    envir$gc <- reticulate::import("gc") # _F I X M E_ rethink
    #
    envir$device <- envir$torch$device(torch_device) # device(type='cuda'); use its $type to test
    # Handle to the eval environ of main Python module:
    envir$py_main <- reticulate::import_main(convert = FALSE) 
    cat("done.\n")
  }
  envir
}

# Available but not used in programming:
.r_to_torch <- function(x, py_handle, device) {
  x <- reticulate::r_to_py(x) # to numpy.ndarray...
  py_handle$py_to_torch(x, device$type)
}
#
## 'memory leak' on GPU If using pure R version:
# r_to_torch <- function(x) {
#   x <- r_to_py(x)
#   x <- x$copy()
#   x <- torch$from_numpy(x)
#   if (device != "cpu") x <- x$to(device)
#   x <- x$float()
#   return(x)
# }

control_py_env <- function(seed=NULL) {
  py_handle <- get_py_MAF_handle()
  if( ! is.null(seed)) {
    abyss <- py_handle$torch$random$manual_seed(as.integer(seed))
  }
  invisible(NULL)
}



