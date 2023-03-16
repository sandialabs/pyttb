"""Higher Order SVD Implementation"""
import warnings
from typing import List, Optional

import numpy as np
import scipy

import pyttb as ttb


def hosvd(
    tensor,
    tol: float,
    verbosity: float = 1,
    dimorder: Optional[List[int]] = None,
    sequential: bool = True,
    ranks: Optional[List[int]] = None,
):
    """Compute sequentially-truncated higher-order SVD (Tucker).

    Computes a Tucker decomposition with relative error
    specified by tol, i.e., it computes a ttensor T such that
    ||X-T||/||X|| <= tol.

    Parameters
    ----------
    tensor: Tensor to factor
    tol: Relative error to stop at
    verbosity: Print level
    dimorder: Order to loop through dimensions
    sequential: Use sequentially-truncated version
    ranks: Specify ranks to consider rather than computing

    Example
    -------
    >>> data = np.array([[29, 39.], [63., 85.]])
    >>> tol = 1e-4
    >>> disable_printing = -1
    >>> tensorInstance = ttb.tensor().from_data(data)
    >>> result = hosvd(tensorInstance, tol, verbosity=disable_printing)
    >>> ((result.full() - tensorInstance).norm() / tensorInstance.norm()) < tol
    True
    """
    # In tucker als this is N
    d = tensor.ndims

    if ranks is not None:
        if len(ranks) != d:
            raise ValueError(
                f"Ranks must be a list of length tensor ndims. Ndims: {d} but got "
                f"ranks: {ranks}."
            )
    else:
        ranks = [0] * d

    # Set up dimorder if not specified (this is copy past from tucker_als
    if not dimorder:
        dimorder = list(range(d))
    else:
        if not isinstance(dimorder, list):
            raise ValueError("Dimorder must be a list")
        elif tuple(range(d)) != tuple(sorted(dimorder)):
            raise ValueError(
                "Dimorder must be a list or permutation of range(tensor.ndims)"
            )

    # TODO should unify printing throughout. Probably easier to use python logging levels
    if verbosity > 0:
        print("Computing HOSVD...\n")

    normxsqr = (tensor**2).collapse()
    eigsumthresh = ((tol**2) * normxsqr) / d

    if verbosity > 2:
        print(
            f"||X||^2 = {normxsqr: g}\n"
            f"tol = {tol: g}\n"
            f"eigenvalue sum threshold = tol^2 ||X||^2 / d = {eigsumthresh: g}"
        )

    # Main Loop
    factor_matrices = [np.empty(1)] * d
    # Copy input tensor, shrinks every step for sequential
    Y = ttb.tensor.from_tensor_type(tensor)

    for k in dimorder:
        # Compute Gram matrix
        Yk = ttb.tenmat.from_tensor_type(Y, np.array([k])).double()
        Z = np.dot(Yk, Yk.transpose())

        # Compute eigenvalue decomposition
        D, V = scipy.linalg.eigh(Z)
        pi = np.argsort(-D, kind="quicksort")
        eigvec = D[pi]

        # If rank not provided compute it.
        if ranks[k] == 0:
            eigsum = np.cumsum(eigvec[::-1])
            eigsum = eigsum[::-1]
            ranks[k] = np.where(eigsum > eigsumthresh)[0][-1]

            if verbosity > 5:
                print(f"Reverse cummulative sum of evals of Gram matrix:")
                for i in range(len(eigsum)):
                    print_msg = f"{i: d}: {eigsum[i]: 6.4f}"
                    if i == ranks[k]:
                        print_msg += " <-- Cutoff"
                    print(print_msg)

        # Extract factor matrix b picking leading eigenvectors of V
        # NOTE: Plus 1 in pi slice for inclusive range to match MATLAB
        factor_matrices[k] = V[:, pi[0 : ranks[k] + 1]]

        # Shrink!
        if sequential:
            Y = Y.ttm(factor_matrices[k].transpose(), k)
    # Extract final core
    if sequential:
        G = Y
    else:
        G = Y.ttm(factor_matrices, transpose=True)

    result = ttb.ttensor.from_data(G, factor_matrices)

    if verbosity > 0:
        diffnormsqr = ((tensor - result.full()) ** 2).collapse()
        relnorm = np.sqrt(diffnormsqr / normxsqr)
        print(f" Size of core: {G.shape}")
        if relnorm <= tol:
            print(f"||X-T||/||X|| = {relnorm: g} <=" f"{tol: f} (tol)")
        else:
            print(
                "Tolerance not satisfied!! "
                f"||X-T||/||X|| = {relnorm: g} >="
                f"{tol: f} (tol)"
            )
            warnings.warn("Specified tolerance was not achieved")
    return result
