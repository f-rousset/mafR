def to_zuko_gmm(nb_components, means, mixture_weights, covariance_matrix, gmm_cov_type, nb_features):
    """
    Convert a GMM (given its parameters) to a zuko GMM object.

    """
    
    weights = py_to_torch(mixture_weights, "cpu")   
    means = py_to_torch(means, "cpu")   
    covs = py_to_torch(covariance_matrix, "cpu")        

    if gmm_cov_type == "full":
        lower = torch.linalg.cholesky(covs)
        diag = torch.diagonal(lower, dim1=-2, dim2=-1).log() # ----------------------------> (K, D)
        tril = lower[(..., *torch.tril_indices(nb_features, nb_features, offset=-1))] # ---> (K, D*(D-1)/2)
        phi = [weights, means, diag, tril]

    elif gmm_cov_type == "diagonal":
        diag = covariance_matrix.log() # --------------------------------------------------> (K, D)
        phi = [weights, means, diag]

    elif gmm_cov_type == "spherical":
        diag = covariance_matrix.log()
        diag = diag[:, None] # ------------------------------------------------------------> (K,)
        phi = [weights, means, diag]

    else:
        raise ValueError(f"Unsupported covariance_type: {gmm_cov_type}")


    zuko_gmm = zuko.mixtures.GMM(nb_features, context=0, components=nb_components, covariance_type=gmm_cov_type)

    with torch.no_grad():
        for i, p in enumerate(phi):
            zuko_gmm.phi[i].copy_(p)

    return zuko_gmm
