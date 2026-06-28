get_py_NLE_handle <- function(envir, reset=FALSE, verbose=TRUE) {
  torch_device <- "cpu"
  if (reset || ! envir$is_set) {
    if (verbose) cat("\nInitializing python session... ") # or 'evaluation' environment.
    # != init sensu-python virtual environment: init_py_env() must have been called.
    NLE_conditional_density_estimation <- sbi_density_estimation <- 
      MAF_predict_cond <- MAF_predict_nocond <- py_to_torch <- NULL

    infile <- system.file('python', "NLE.py", package='mafR')
    chk <- try(reticulate::source_python(infile)) # this provides objects in the present R closure!
    if (inherits(chk,"try-error")) {
      message("you need a properly set up python environment to use 'mafR': cf. init_py_env().")
      return(attr(chk,"condition")$message)
    } else {
      # objects are no longer NULL
      envir$NLE_conditional_density_estimation <- NLE_conditional_density_estimation
      envir$sbi_density_estimation <- sbi_density_estimation
      envir$MAF_predict_cond <- MAF_predict_cond
      envir$MAF_predict_nocond <- MAF_predict_nocond
      envir$py_to_torch <- py_to_torch
      envir$is_set <- TRUE
      #
      ## Python packages to be called from R
      torch <- envir$torch <- reticulate::import("torch")
      envir$device <- torch$device(torch_device) # device(type='cuda') or 'mps'; use its $type to test
      # Handle to the eval environ of main Python module:
      envir$py_main <- reticulate::import_main(convert = FALSE) # cf Infusion sources for its use
      if (verbose) cat("done.\n")
      envir
    }
  }
  envir
}
