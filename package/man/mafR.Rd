\name{mafR}
\alias{mafR}
\alias{mafR-package}

\docType{package}

\title{Interface for masked autoregressive flows}

\description{This wraps Python procedures to train Masked Autoregressive Flows (MAFs, Paramakarios et al. 2017) using the Python package \code{zuko}. It has been tested with version 1.1.0 and 1.2.0 of that package. Note that objects created by its version 1.2.0 cannot be read with its version 1.1.0 (i.e., when saved in and read from \code{pickle} files).

The simplest portable way to get \pkg{mafR} working may be to install it in a conda environment. Below is a complete installation recipe. More information about alternative installation procedure may be found on the Git repository for \pkg{mafR}, \url{https://github.com/f-rousset/mafR}.

\preformatted{
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
conda create --name maf-conda python==3.10
conda activate maf-conda

pip install zuko

conda install R                                       
conda install conda-forge::r-gmp
conda install conda-forge::gsl
}
and, in an R session within the \code{maf-conda} environment:
\preformatted{
install.packages("reticulate") 
library(reticulate)
use_condaenv(condaenv="maf-conda", conda="~/miniconda3/bin/conda")
install.packages("mafR")

# 'mafR' was first designed for use with 'Infusion':
install.packages("Infusion")
install.packages("Rmixmod") # only a Suggested dependency of Infusion, but needed.
}
}

\references{
Papamakarios, G., D. Sterratt, and I. Murray. 2019. Sequential Neural Likelihood:
Fast Likelihood-free Inference with Autoregressive Flows. Proceedings of the 
Twenty-Second International Conference on Artificial Intelligence and Statistics, 
PMLR 89:837-848, 2019.
https://doi.org/10.48550/arXiv.1705.07057 ; 
https://proceedings.mlr.press/v89/papamakarios19a.html

Rozet, F., Divo, F., Schnake, S (2023) Zuko: Normalizing flows in PyTorch. 
https://doi.org/10.5281/zenodo.7625672
}

