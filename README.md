This package allows one to call from R the implementation of Masked Autoregressive Flows (MAFs) in the `zuko` python package ([Rozet et al. 2023](dx.doi.org/10.5281/zenodo.7625672)).
It has been designed more specifically for use with the [`Infusion` R package](gitlab.mbb.univ-montp2.fr/francois/Infusion).
Although the `mafR` package is trivial to install, you will need to install a Python environment with appropriate packages in order to be able to use `mafR`.
See the installation instructions in the main documentation page for `mafR`. 

Then, you should be able to call MAFs when using `Infusion` for simulation-based inference, as illustrated by the following toy example:

##### Preliminaries:
```
  library(Infusion)
  cat(cli::col_yellow("(!) This example uses cuda (!)"))
  config_mafR(torch_device="cuda")
```
##### Define function for sampling from 3-parameter gaussian mixture:
```
  myrnorm3 <- function(mu1,mu2,s2,sample.size=40L) {
    sam1 <- rnorm(n=sample.size,mean=mu1,sd=sqrt(s2))
    sam2 <- rnorm(n=sample.size,mean=mu2,sd=sqrt(s2))
    e_mu1 <- mean(sam1)
    e_mu2 <- mean(sam2)
    e_s2 <- mean(var(sam1),var(sam2))
    return(c(mean1=e_mu1,mean2=e_mu2,var=e_s2))
  } 
```
##### Simulate data, standing for the data to be analyzed:
```
  set.seed(123)
  Sobs <- myrnorm3(mu1=4,mu2=2,s2=1) 
```
##### Build initial reference table of simulated samples:
```
  parsp <- init_reftable(lower=c(mu1=2.8,mu2=1,s2=0.2), 
                         upper=c(mu1=5.2,mu2=3,s2=3))
  parsp <- cbind(parsp)
  simuls <- add_reftable(Simulate="myrnorm3", parsTable=parsp)
```
##### Perform initial inference of likelihood surface (note  using="mafR"):
```  
  densv <- infer_SLik_joint(simuls,stat.obs=Sobs, using="mafR")  
```
##### Carry on iterative workflow using inferred surface:
```
  slik_j <- MSL(densv, eval_RMSEs = FALSE) ## find the maximum of the log-likelihood surface
  slik_j <- refine(slik_j,maxit=5, update_projectors=TRUE) ## further steps as in a standard Infusion workflow
```
