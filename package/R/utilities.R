.get_GPU_mem <- function(GPU_mem, torch_device, get_gpu_info) {
  if (length(grep("cuda",torch_device))) {
    GPU_mem <- get_gpu_info("cuda") # for current device: available and total dedicated GPU memory
    ## => This appeared to fail but only bc R handles only 32-bit ints.
  } else if (length(grep("mps",torch_device))) {
    GPU_mem <- get_gpu_info("mps") # for current device: available and total dedicated GPU memory
  } else {
    if (missing(GPU_mem) || is.null(GPU_mem)) stop("Please specify 'GPU_mem'.")
    if (length(GPU_mem)==1L) GPU_mem <- c(NA_real_,GPU_mem) 
  }
  unlist(GPU_mem)
}

get_py_MAF_handle <- function(envir, reset=FALSE, torch_device="cpu",GPU_mem=NULL) {
  if (reset || ! envir$is_set) {
    cat("\nInitializing python session... ")
    MAF_density_estimation <- MAF_conditional_density_estimation <- 
      MAF_predict_cond <- MAF_predict_nocond <- MAF_simulate_cond <- 
      MAF_transform <- py_to_torch <- get_gpu_info <- NULL
    # reticulate::source_python(paste0(Infusion::projpath(),"/../MAF-R/MAF.py"))
    infile <- system.file('python', "MAF.py", package='mafR')
    reticulate::source_python(infile)
    envir$MAF_density_estimation <- MAF_density_estimation
    envir$MAF_conditional_density_estimation <- MAF_conditional_density_estimation
    envir$MAF_predict_cond <- MAF_predict_cond
    envir$MAF_predict_nocond <- MAF_predict_nocond
    envir$MAF_simulate_cond <- MAF_simulate_cond
    envir$MAF_transform <- MAF_transform
    envir$py_to_torch <- py_to_torch
    # the Python source has also provided get_gpu_info(), used below only.
    envir$is_set <- TRUE
    ## Python packages to be called from R
    torch <- envir$torch <- reticulate::import("torch")
    # envir$gc <- reticulate::import("gc") 
    #
    envir$device <- torch$device(torch_device) # device(type='cuda') or 'mps'; use its $type to test
    if (is.null(GPU_mem) && torch_device != "cpu") envir$gpu_memory <- 
      .get_GPU_mem(GPU_mem, torch_device, get_gpu_info)
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

control_py_env <- function(py_handle, seed=NULL) {
  if( ! is.null(seed)) {
    abyss <- py_handle$torch$random$manual_seed(as.integer(seed))
  }
  invisible(NULL)
}



