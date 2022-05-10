# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np


def khatrirao(*listOfMatrices, reverse=False):
    """
    KHATRIRAO Khatri-Rao product of matrices.

    KHATRIRAO(A,B) computes the Khatri-Rao product of matrices A and
    B that have the same number of columns.  The result is the
    column-wise Kronecker product
    [KRON(A(:,1),B(:,1)) ... KRON(A(:,n),B(:,n))]

    Parameters
    ----------
    Matrices: [:class:`numpy.ndarray`] or :class:`numpy.ndarray`,:class:`numpy.ndarray`...
    reverse: bool Set to true to calculate product in reverse

    Returns
    -------
    product: float

    Examples
    --------
    >>>A = np.random.norm(size=(5,2))
    >>>khatrirao(A,B) #<-- Khatri-Rao of A and B
    >>>>khatrirao(B,A,reverse=True) #<-- same thing as above
    >>>>khatrirao([A,A,B]) #<-- passing a list
    >>>>khatrirao([B,A,A},reverse = True) #<-- same as above
    """
    #Determine if list of matrices of multiple matrix arguments
    if isinstance(listOfMatrices[0], list):
        if len(listOfMatrices) == 1:
            listOfMatrices = listOfMatrices[0]
        else:
            assert False, "Khatri Rao Acts on multiple Array arguments or a list of Arrays"

    # Error checking on input and set matrix order
    if reverse == True:
        listOfMatrices = list(reversed(listOfMatrices))
    ndimsA = [len(matrix.shape) == 2 for matrix in listOfMatrices]
    if not np.all(ndimsA):
        assert False, 'Each argument must be a matrix'

    ncolFirst = listOfMatrices[0].shape[1]
    ncols = [matrix.shape[1] == ncolFirst for matrix in listOfMatrices]
    if not np.all(ncols):
        assert False, 'All matrices must have the same number of columns.'

    # Computation
    P = listOfMatrices[0]
    if ncolFirst == 1:
        for i in listOfMatrices[1:]:
            P = np.reshape(i, newshape=(ncolFirst, -1))*np.reshape(P, newshape=(-1, ncolFirst))
    else:
        for i in listOfMatrices[1:]:
            P = np.reshape(i, newshape=(-1, 1, ncolFirst))*np.reshape(P, newshape=(1, -1, ncolFirst))

    return np.reshape(P, newshape=(-1, ncolFirst))
