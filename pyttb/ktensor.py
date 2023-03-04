# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

"""Classes and functions for working with Kruskal tensors."""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

import pyttb as ttb
from pyttb.pyttb_utils import *


class ktensor(object):
    """
    Class for Kruskal tensors (decomposed).

    Contains the following data members:

      * ``weights``: :class:`numpy.ndarray` vector containing the weights of the rank-1 tensors defined by the outer \
    products of the column vectors of the factor_matrices.

      * ``factor_matrices``: :class:`list` of :class:`numpy.ndarray`. The length of the list is equal to the number \
    of dimensions of the tensor. The shape of the ith element of the list is (n_i, r), where n_i is the length \
    dimension i and r is the rank of the tensor (as well as the length of the weights vector).

    Although the constructor, `__init__()`, can be used to create an empty :class:`pyttb.ktensor`, there
    are several class methods that can be used to create an instance of this class:

      * :meth:`from_data`
      * :meth:`from_tensor_type`
      * :meth:`from_factor_matrices`
      * :meth:`from_function`
      * :meth:`from_vector`
    """

    __slots__ = ("weights", "factor_matrices")

    def __init__(self):
        """
        Constructor for :class:`pyttb.ktensor`

        The constructor takes no arguments and returns an empty :class:`pyttb.ktensor`.
        """
        # Empty constructor
        self.weights = np.array([])  # renamed from lambda to weights
        self.factor_matrices = (
            []
        )  # changed from cell array to list; changed name from 'u' to 'factor_matrices'

    @classmethod
    def from_data(cls, weights, *factor_matrices):
        """
        Construct a :class:`pyttb.ktensor` from weights and factor matrices.

        The length of the list or the number of arguments specified by `factor_matrices` must equal
        the length of `weights`

        Parameters
        ----------
        weights: :class:`numpy.ndarray`
        factor_matrices: :class:`list` of :class:`numpy.ndarray` or variable number of :class:`numpy.ndarray`

        Returns
        -------
        :class:`pyttb.ktensor`

        Example
        -------
        Create a `ktensor` from weights and a list of factor matrices:

        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> K_from_list = ttb.ktensor.from_data(weights, [fm0, fm1])
        >>> print(K_from_list)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Create a `ktensor` from weights and a factor matrices passed as arguments:

        >>> K_from_args = ttb.ktensor.from_data(weights, fm0, fm1)
        >>> print(K_from_args)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]
        """
        # Check individual input parameters
        assert isinstance(weights, np.ndarray) and isvector(
            weights
        ), "Input parameter 'weights' must be a numpy.array type vector."

        # Input can be a list or sequences of factor_matrices
        if isinstance(factor_matrices[0], list):
            _factor_matrices = [f.copy() for f in factor_matrices[0]]
        else:
            _factor_matrices = [f.copy() for f in factor_matrices[0:]]
        for fm in _factor_matrices:
            assert isinstance(
                fm, np.ndarray
            ), "Input parameter 'factor_matrices' must be a list of numpy.array's."

        # Check dimensions of weights and factor_matrices
        num_weights = len(weights)
        for i, fm in enumerate(_factor_matrices):
            assert (
                num_weights == fm.shape[1]
            ), "Size of factor_matrix {} does not match number of weights ({}).".format(
                i, num_weights
            )

        # Create ktensor and populate data members
        k = cls()
        k.weights = weights.copy()
        if k.weights.dtype != float:
            print("converting weights from {} to float".format(k.weights.dtype))
            k.weights = k.weights.astype(float)
        k.factor_matrices = _factor_matrices
        for i in range(len(k.factor_matrices)):
            if k.factor_matrices[i].dtype != float:
                print(
                    "converting factor_matrices[{}] from {} to float".format(
                        i, k.factor_matrices[i].dtype
                    )
                )
                k.factor_matrices[i] = k.factor_matrices[i].astype(float)

        return k

    @classmethod
    def from_tensor_type(cls, source) -> ktensor:
        """
        Construct a ktensor from another ktensor. A deep copy of the data from the input ktensor
        is used for the new ktensor.

        Parameters
        ----------
        source: :class:`pyttb.ktensor`

        Returns
        -------
        :class:`pyttb.ktensor`

        Example
        -------
        Create an instance of a `ktensor`:

        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> factor_matrices = [fm0, fm1]
        >>> K_source = ttb.ktensor.from_factor_matrices(factor_matrices)
        >>> print(K_source)
        ktensor of shape 2 x 2
        weights=[1. 1.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Create another instance of a `ktensor` from the original one above:

        >>> K_copy = ttb.ktensor.from_tensor_type(K_source)
        >>> print(K_copy)
        ktensor of shape 2 x 2
        weights=[1. 1.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

         See also :func:`~pyttb.ktensor.copy`
        """
        if isinstance(source, ktensor):
            return cls().from_data(
                source.weights.copy(), [f.copy() for f in source.factor_matrices]
            )

        # TODO impement conversion when symktensor has been implemented
        # if isinstance(source, ttb.symktensor):
        #     #    self.weights = args[0].weights;
        #     #    # MATLAB=> [t.u{1:varargin{1}.m,1}] = deal(varargin{1}.u);
        #     #    return
        #     raise NotImplementedError

        assert False, "Cannot convert from {} to ktensor".format(str(source.__class__))

    @classmethod
    def from_factor_matrices(cls, *factor_matrices):
        """
        Construct a ktensor from factor matrices. The weights of the returned ktensor will all be equal to 1.

        Parameters
        ----------
        factor_matrices: :class:`list` of :class:`numpy.ndarray` or variable number of :class:`numpy.ndarray`
            The number of columns of each of the factor matrices must be the same.

        Returns
        -------
        :class:`pyttb.ktensor`

        Example
        -------
        Create a `ktensor` from a list of factor matrices:

        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> factor_matrices = [fm0, fm1]
        >>> K_from_list = ttb.ktensor.from_factor_matrices(factor_matrices)
        >>> print(K_from_list)
        ktensor of shape 2 x 2
        weights=[1. 1.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Create a `ktensor` from a sequence of factor matrices passed as arguments:

        >>> K_from_args = ttb.ktensor.from_factor_matrices(fm0, fm1)
        >>> print(K_from_args)
        ktensor of shape 2 x 2
        weights=[1. 1.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]
        """
        # CONSTRUCTOR FROM LIST OF FACTOR MATRICES
        # Input can be a list or sequences of factor_matrices
        if isinstance(factor_matrices[0], list):
            _factor_matrices = factor_matrices[0]
        else:
            _factor_matrices = [f for f in factor_matrices[0:]]
        for fm in _factor_matrices:
            assert isinstance(
                fm, np.ndarray
            ), "Input parameter 'factor_matrices' must be a list of numpy.array's."
        nc = _factor_matrices[0].shape[1]
        return cls().from_data(np.ones(nc), _factor_matrices)

    @classmethod
    def from_function(cls, fun, shape, num_components):
        """
        Construct a ktensor whose factor matrix entries are set using a function. The weights of the returned ktensor will all be equal to 1.

        Parameters
        ----------
        fun: function
          A function that can accept a shape (i.e., :class:`tuple` of dimension sizes) and return a :class:`numpy.ndarray`
          of that shape. Example functions include `numpy.random.random_sample`, `numpy,zeros`, `numpy.ones`.
        shape: :class:`tuple`
        num_components: int

        Returns
        -------
        :class:`pyttb.ktensor`

        Example
        -------
        Create a `ktensor` with entries of the factor matrices taken from a uniform random distribution:

        >>> np.random.seed(1)
        >>> K_random = ttb.ktensor.from_function(np.random.random_sample, (2, 3, 4), 2)
        >>> print(K_random)  # doctest: +ELLIPSIS
        ktensor of shape 2 x 3 x 4
        weights=[1. 1.]
        factor_matrices[0] =
        [[4.1702...e-01 7.2032...e-01]
         [1.1437...e-04 3.0233...e-01]]
        factor_matrices[1] =
        [[0.1467... 0.0923...]
         [0.1862... 0.3455...]
         [0.3967... 0.5388...]]
        factor_matrices[2] =
        [[0.4191... 0.6852...]
         [0.2044... 0.8781...]
         [0.0273... 0.6704...]
         [0.4173...  0.5586...]]

        Create a `ktensor` with entries equal to 1:

        >>> K_ones = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> print(K_ones)
        ktensor of shape 2 x 3 x 4
        weights=[1. 1.]
        factor_matrices[0] =
        [[1. 1.]
         [1. 1.]]
        factor_matrices[1] =
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
        factor_matrices[2] =
        [[1. 1.]
         [1. 1.]
         [1. 1.]
         [1. 1.]]

        Create a `ktensor` with entries equal to 0:

        >>> K_zeros = ttb.ktensor.from_function(np.zeros, (2, 3, 4), 2)
        >>> print(K_zeros)
        ktensor of shape 2 x 3 x 4
        weights=[1. 1.]
        factor_matrices[0] =
        [[0. 0.]
         [0. 0.]]
        factor_matrices[1] =
        [[0. 0.]
         [0. 0.]
         [0. 0.]]
        factor_matrices[2] =
        [[0. 0.]
         [0. 0.]
         [0. 0.]
         [0. 0.]]
        """
        # CONSTRUCTOR FROM FUNCTION HANDLE
        assert callable(fun), "Input parameter 'fun' must be a function."
        assert isinstance(shape, tuple), "Input parameter 'shape' must be a tuple."
        assert isinstance(
            num_components, int
        ), "Input parameter 'num_components' must be an int."
        nd = len(shape)
        weights = np.ones(num_components)
        factor_matrices = []
        for i in range(nd):
            factor_matrices.append(fun((shape[i], num_components)))
        return cls().from_data(weights, factor_matrices)

    @classmethod
    def from_vector(cls, data, shape, contains_weights):
        """
        Construct a ktensor from a vector. The rank of the `ktensor` is inferred from the shape and size of the vector.

        Parameters
        ----------
        data: :class:`numpy.ndarray`
          Vector containing either elements of the factor matrices (when contains_weights==False) or
          elements of the weights and factor matrices (when contains_weights==True). When both the
          elements of the weights and the factor_matrices are present, the weights come first and
          the columns of the factor matrices come next.
        shape: :class:`numpy.ndarray`
          Vector containing the shape of the tensor (i.e., sizes of the dimensions).
        contains_weights: bool
          Flag to specify whether or not ``data`` contains weights. If False, all weights are set to 1.

        Returns
        -------
        :class:`pyttb.ktensor`

        Example
        -------
        Create a `ktensor` from a vector containing only elements of the factor matrices:

        >>> rank = 2
        >>> shape = np.array([2, 3, 4])
        >>> data = np.arange(1, rank*sum(shape)+1).astype(float)
        >>> K_without_weights = ttb.ktensor.from_vector(data[:], shape, False)
        >>> print(K_without_weights)
        ktensor of shape 2 x 3 x 4
        weights=[1. 1.]
        factor_matrices[0] =
        [[1. 3.]
         [2. 4.]]
        factor_matrices[1] =
        [[ 5.  8.]
         [ 6.  9.]
         [ 7. 10.]]
        factor_matrices[2] =
        [[11. 15.]
         [12. 16.]
         [13. 17.]
         [14. 18.]]

        Create a `ktensor` from a vector containing elements of both the weights and the factor matrices:

        >>> weights = 2 * np.ones(rank).astype(float)
        >>> weights_and_data = np.concatenate((weights, data), axis=0)
        >>> K_with_weights = ttb.ktensor.from_vector(weights_and_data[:], shape, True)
        >>> print(K_with_weights)
        ktensor of shape 2 x 3 x 4
        weights=[2. 2.]
        factor_matrices[0] =
        [[1. 3.]
         [2. 4.]]
        factor_matrices[1] =
        [[ 5.  8.]
         [ 6.  9.]
         [ 7. 10.]]
        factor_matrices[2] =
        [[11. 15.]
         [12. 16.]
         [13. 17.]
         [14. 18.]]
        """
        assert isvector(data), "Input parameter 'data' must be a numpy.array vector."
        assert isinstance(
            shape, np.ndarray
        ), "Input parameter 'shape' must be a numpy.array vector."
        assert isinstance(
            contains_weights, bool
        ), "Input parameter 'contains_weights' must be a bool."

        if isrow(data):
            data = data.T

        # compute the number of components from inputs
        if contains_weights:
            num_components = len(data) / (sum(shape) + 1)
        else:
            num_components = len(data) / sum(shape)

        if round(num_components) != num_components:
            assert False, "Input parameter 'data' is not the right length."
        else:
            num_components = int(num_components)

        # extract weights from input vector if present
        if contains_weights:
            weights = data[0:num_components].copy()
            shift = num_components
        else:
            weights = np.ones(num_components)
            shift = 0

        # extract factor matrices
        factor_matrices = []
        for n in range(len(shape)):
            mstart = num_components * sum(shape[0:n]) + shift
            mend = num_components * sum(shape[0 : n + 1]) + shift
            # the following will match MATLAB output
            factor_matrix = np.reshape(
                data[mstart:mend].copy(), (shape[n], num_components), order="F"
            )
            factor_matrices.append(factor_matrix)

        return cls().from_data(weights, factor_matrices)

    def arrange(self, weight_factor=None, permutation=None):
        """
        Arrange the rank-1 components of a `ktensor`. The columns are permuted in place, so you must make a copy
        before calling this method if you want to store the original.

        Parameters
        ----------
        weight_factor:
        permutation:

        Returns
        -------
        :class:`pyttb.ktensor`

        Example
        -------
        Create the initial `ktensor`:

        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> K = ttb.ktensor.from_data(weights, [fm0, fm1])
        >>> print(K)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Arrange the columns of the factor matrices using a permutation:

        >>> p = [1,0]
        >>> K.arrange(permutation=p)
        >>> print(K)
        ktensor of shape 2 x 2
        weights=[2. 1.]
        factor_matrices[0] =
        [[2. 1.]
         [4. 3.]]
        factor_matrices[1] =
        [[6. 5.]
         [8. 7.]]
        """

        if permutation is not None and weight_factor is not None:
            assert (
                False
            ), "Weighting and permuting the ktensor at the same time is not allowed."

        if permutation is not None and isinstance(
            permutation, (tuple, list, np.ndarray)
        ):
            if len(permutation) == self.ncomponents:
                self.weights = self.weights[permutation]
                for i in range(self.ndims):
                    self.factor_matrices[i] = self.factor_matrices[i][:, permutation]
                return
            else:
                assert (
                    False
                ), "Number of elements in permutation does not match number of components in ktensor."

        # TODO there is a relationship here between normalize and arrange that repeats tasks. Can this be made to be more efficient?
        # ensure that factor matrices are normalized
        self.normalize()

        # sort
        p = np.argsort(self.weights)[::-1]
        self.weights = self.weights[p]
        for i in range(self.ndims):
            self.factor_matrices[i] = self.factor_matrices[i][:, p]

        # TODO is this necessary? Matlab only absorbs into the last factor, not factor N as is documented
        # absorb weight into one factor if requested
        if weight_factor is not None:
            pass
        # if exist('foo','var')
        #     r = length(X.lambda);
        #     X.u{end} = full(X.u{end} * spdiags(X.lambda,0,r,r));
        #     X.lambda = ones(size(X.lambda));
        # end
        return self

    def copy(self):
        """
        Make a deep copy of a `ktensor`.

        Returns
        -------
        :class:`pyttb.ktensor`

        Example
        -------
        Create a random `ktensor` with weights of 1:

        >>> np.random.seed(1)
        >>> K = ttb.ktensor.from_function(np.random.random_sample, (2, 3, 4), 2)
        >>> print(K)  # doctest: +ELLIPSIS
        ktensor of shape 2 x 3 x 4
        weights=[1. 1.]
        factor_matrices[0] =
        [[4.1702...e-01 7.2032...e-01]
         [1.1437...e-04 3.0233...e-01]]
        factor_matrices[1] =
        [[0.1467... 0.0923...]
         [0.1862... 0.3455...]
         [0.3967... 0.5388...]]
        factor_matrices[2] =
        [[0.4191... 0.6852...]
         [0.2044... 0.8781...]
         [0.0273... 0.6704...]
         [0.4173... 0.5586...]]

        Create a copy of the `ktensor` and change the weights:

        >>> K1 = K.copy()
        >>> K1.weights = np.array([2., 3.])
        >>> print(K1)  # doctest: +ELLIPSIS
        ktensor of shape 2 x 3 x 4
        weights=[2. 3.]
        factor_matrices[0] =
        [[4.1702...e-01 7.2032...e-01]
         [1.1437...e-04 3.023...e-01]]
        factor_matrices[1] =
        [[0.1467... 0.0923...]
         [0.1862... 0.3455...]
         [0.3967... 0.5388...]]
        factor_matrices[2] =
        [[0.4191... 0.6852...]
         [0.2044... 0.8781...]
         [0.0273... 0.6704...]
         [0.4173... 0.5586...]]
        """
        return ttb.ktensor.from_tensor_type(self)

    def double(self):
        """
        Convert `ktensor` to numpy array.

        Returns
        -------
        :class:`numpy.ndarray`

        Example
        -------
        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor.from_data(weights, factor_matrices)
        >>> K.double()
        array([[29., 39.],
               [63., 85.]])
        """
        return self.full().double()

    def end(self, k=None):
        """
        Last index of indexing expression for `ktensor`.

        Parameters
        ----------
        k: int
            dimension for subscripted indexing
        n: int
            dimensions to index

        Returns
        -------
        int: index

        Example
        -------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> print(K.end(2))
        3
        """

        if k is not None:  # Subscripted indexing
            return self.shape[k] - 1
        else:  # For linear indexing
            return np.prod(self.shape) - 1

    def extract(self, idx=None):
        """
        Creates a new `ktensor` with only the specified components.

        Parameters
        ----------
        idx: int, tuple, list, :class:`numpy.ndarray`
            Index set of components to extract. It should be the case that `idx` is a subset of
            [0,...,self.ncomponents]. If this parameter is None or is empty, a copy of the ktensor is returned.

        Returns
        -------
        :class:`ktensor`

        Example
        -------
        Create a `ktensor`:

        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> K = ttb.ktensor.from_data(weights, [fm0, fm1])
        >>> print(K)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Create a new `ktensor`, extracting only the second component from each factor of the original `ktensor`:

        >>> K.extract([1])
        ktensor of shape 2 x 2
        weights=[2.]
        factor_matrices[0] =
        [[2.]
         [4.]]
        factor_matrices[1] =
        [[6.]
         [8.]]
        """
        # return a copy if no components have been specified
        if idx is None:
            return self.from_tensor_type(self)

        if isinstance(idx, (int, tuple, list, np.ndarray)):
            if isinstance(idx, int):
                components = [idx]
            else:
                components = idx
            if len(components) == 0 or len(components) > self.ncomponents:
                assert (
                    False
                ), "Number of components requested is not valid: {} (should be in [1,...,{}]).".format(
                    len(components), self.ncomponents
                )
            else:
                # check that all requested component indices are valid
                invalid_entries = []
                for i in range(len(components)):
                    if components[i] not in range(self.ncomponents):
                        invalid_entries.append(components[i])
                if len(invalid_entries) > 0:
                    assert (
                        False
                    ), "Invalid component indices to be extracted: {} not in range({})".format(
                        str(invalid_entries), self.ncomponents
                    )
                new_weights = self.weights[components]
                new_factor_matrices = []
                for i in range(self.ndims):
                    new_factor_matrices.append(self.factor_matrices[i][:, components])
                return self.from_data(new_weights, new_factor_matrices)
        else:
            assert False, "Input parameter must be an int, tuple, list or numpy.ndarray"

    def fixsigns(self, other=None):
        """
        Change the elements of a `ktensor` in place so that the largest magnitude entries for
        each column vector in each factor matrix are positive, provided that the
        sign on pairs of vectors in a rank-1 component can be flipped.

        Parameters
        ----------
        other: :class:`pyttb.ktensor`
            If not None, returns a version of the `ktensor` where some of the signs of
            the columns of the factor matrices have been flipped to better align
            with `other`.

        Returns
        -------
        :class:`pyttb.ktensor`
            The changes are made in place and a reference to the updated tensor is returned

        Example
        -------
        Create a `ktensor` with negative large magnitude entries:

        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> K = ttb.ktensor.from_data(weights, [fm0, fm1])
        >>> K.factor_matrices[0][1, 1] = -K.factor_matrices[0][1, 1]
        >>> K.factor_matrices[1][1, 1] = -K.factor_matrices[1][1, 1]
        >>> print(K)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[ 1.  2.]
         [ 3. -4.]]
        factor_matrices[1] =
        [[ 5.  6.]
         [ 7. -8.]]

        Fix the signs of the largest magnitude entries:

        >>> print(K.fixsigns())
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[ 1. -2.]
         [ 3.  4.]]
        factor_matrices[1] =
        [[ 5. -6.]
         [ 7.  8.]]

        Fix the signs using another `ktensor`:

        >>> K = ttb.ktensor.from_data(weights, [fm0, fm1])
        >>> K2 = K.copy()
        >>> K2.factor_matrices[0][1, 1] = - K2.factor_matrices[0][1, 1]
        >>> K2.factor_matrices[1][1, 1] = - K2.factor_matrices[1][1, 1]
        >>> K = K.fixsigns(K2)
        >>> print(K)  # doctest: +ELLIPSIS
        ktensor of shape 2 x 2
        weights=[27.2029... 89.4427...]
        factor_matrices[0] =
        [[ 0.3162... -0.4472...]
         [ 0.9486... -0.8944...]]
        factor_matrices[1] =
        [[ 0.5812... -0.6...]
         [ 0.8137... -0.8...]]
        """
        if other == None:
            for r in range(self.ncomponents):
                sgn = np.zeros(self.ndims)
                for n in range(self.ndims):
                    index_max = np.argmax(abs(self.factor_matrices[n][:, r]))
                    sgn[n] = np.sign(self.factor_matrices[n][index_max, r])

                negidx = np.nonzero(sgn == -1)[0]
                nflip = int(2 * np.floor(np.size(negidx) / 2))

                for i in range(nflip):
                    n = negidx[i]
                    self.factor_matrices[n][:, r] = -self.factor_matrices[n][:, r]

            return self
        else:
            if not isinstance(other, ktensor):
                assert False, "other must be a ktensor"

            self.normalize()
            other = other.normalize()

            N = self.ndims
            RA = self.ncomponents
            RB = other.ncomponents

            # Try to fix the signs for each component
            best_sign = np.zeros((N, RA))
            for r in range(RB):
                # Compute the inner products. They should mostly be O(1) if there is a
                # good match because the factors have prevsiouly been normalized. If
                # the signs are correct, then the score should be +1. Otherwise we need
                # to flip the sign and the score should be -1.
                sgn_score = np.zeros(N)
                for n in range(N):
                    sgn_score[n] = (
                        self.factor_matrices[n][:, r].T @ other.factor_matrices[n][:, r]
                    )

                # Sort the sign scores.
                sort_idx = np.argsort(sgn_score)
                sort_sgn_score = sgn_score.copy()[sort_idx]

                # Determine the number of scores that should be flipped.
                breakpt = np.nonzero(sort_sgn_score < 0)[-1]

                # If nothing needs to be flipped, then move on the the next component.
                if len(breakpt) == 0:
                    continue
                else:
                    breakpt = breakpt[-1]

                # Need to flip signs in pairs. If we don't have an even number of
                # negative sign scores, then we need to decide to do one fewer or one
                # more.
                if np.mod(breakpt + 1, 2) == 0:
                    endpt = breakpt + 1
                else:
                    warnings.warn("Trouble fixing signs for mode {}".format(r))
                    if (breakpt < RB) and (
                        -sort_sgn_score[breakpt] > sort_sgn_score[breakpt + 1]
                    ):
                        endpt = breakpt + 1
                    else:
                        endpt = breakpt - 1

                # Flip the signs
                for i in range(endpt):
                    self.factor_matrices[sort_idx[i]][:, r] = (
                        -1 * self.factor_matrices[sort_idx[i]][:, r]
                    )
                    best_sign[sort_idx[i], r] = -1

            return self

    def full(self):
        """
        Convert a `ktensor` to a (dense) `tensor`.

        Returns
        -------
        :class:`pyttb.tensor`

        Example
        -------
        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> K = ttb.ktensor.from_data(weights, [fm0, fm1])
        >>> print(K.full()) # doctest: +NORMALIZE_WHITESPACE
        tensor of shape 2 x 2
        data[:, :] =
        [[29. 39.]
         [63. 85.]]
        <BLANKLINE>
        """
        data = self.weights @ ttb.khatrirao(self.factor_matrices, reverse=True).T
        return ttb.tensor.from_data(data, self.shape)

    def innerprod(self, other):
        """
        Efficient inner product with a `ktensor`.

        Efficiently computes the inner product between two tensors, self and other.  If other is a ktensor,
        the inner product is computed using inner products of the factor matrices. Otherwise, the inner product
        is computed using ttv with all of the columns of self's factor matrices.

        Parameters
        ----------
        other: compute inner product of ktensor with other

        Returns
        -------
        :float

        Example
        -------
        >>> K = ttb.ktensor.from_function(np.ones, (2,3,4), 2)
        >>> print(K.innerprod(K))
        96.0
        """
        if not (self.shape == other.shape):
            assert False, "Innerprod can only be computed for tensors of the same size"

        if isinstance(other, ktensor):
            M = np.outer(self.weights, other.weights)
            for i in range(self.ndims):
                M = M * (self.factor_matrices[i].T @ other.factor_matrices[i])
            return np.sum(np.sum(M))

        if isinstance(other, (ttb.tensor, ttb.sptensor, ttb.ttensor)):
            res = 0.0
            for r in range(self.ncomponents):
                vecs = []
                for n in range(self.ndims):
                    vecs.append(self.factor_matrices[n][:, r])
                res = res + self.weights[r] * other.ttv(vecs)
            return res

    def isequal(self, other):
        """
        Equal comparator for `ktensors`.

        Parameters
        ----------
        other: compare equality of ktensor to other

        Returns
        -------
        :bool

        Example
        -------
        >>> K1 = ttb.ktensor.from_function(np.ones, (2,3,4), 2)
        >>> weights = np.ones((2, 1))
        >>> factor_matrices = [np.ones((2, 2)), np.ones((3, 2)), np.ones((4, 2))]
        >>> K2 = ttb.ktensor.from_data(weights, factor_matrices)
        >>> print(K1.isequal(K2))
        True
        """
        if not isinstance(other, ktensor):
            return False
        if not (self.ncomponents == other.ncomponents):
            return False
        if not (self.weights == other.weights).all():
            return False
        for k in range(self.ndims):
            if not (self.factor_matrices[k] == other.factor_matrices[k]).all():
                return False
        return True

    def issymmetric(self, return_diffs=False):
        """
        Returns true if the `ktensor` is exactly symmetric for every permutation.

        Parameters
        ----------
        return_diffs: bool
            If True, returns the matrix of the norm of the differences between the factor matrices

        Returns
        -------
        :bool
        :class:`numpy.ndarray`
            Matrix of the norm of the differences between the factor matrices (optional)

        Example
        -------
        Create a 'ktensor` that is symmetric and test if it is symmetric:

        >>> K = ttb.ktensor.from_function(np.ones, (3, 3, 3), 2)
        >>> print(K.issymmetric())
        True

        Create a `ktensor` that is not symmetric and return the differences:

        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> K1 = ttb.ktensor.from_data(weights, [fm0, fm1])
        >>> issym, diffs = K1.issymmetric(return_diffs=True)
        >>> print(diffs)
        [[0. 8.]
         [0. 0.]]
        """
        diffs = np.zeros((self.ndims, self.ndims))
        for i in range(self.ndims):
            for j in range(i + 1, self.ndims):
                if not (self.factor_matrices[i].shape == self.factor_matrices[j].shape):
                    diffs[i, j] = np.inf
                elif (self.factor_matrices[i] == self.factor_matrices[j]).all():
                    diffs[i, j] = 0
                else:
                    diffs[i, j] = np.linalg.norm(
                        self.factor_matrices[i] - self.factor_matrices[j]
                    )
        issym = (diffs == 0).all()

        if return_diffs:
            return issym, diffs
        else:
            return issym

    def mask(self, W):
        """
        Extract `ktensor` values as specified by a mask `tensor`

        Parameters
        ----------
        W: :class:`pyttb.sptensor`

        Returns
        -------
        :class:`Numpy.ndarray`

        Example
        -------
        Create a `ktensor`:

        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> K = ttb.ktensor.from_data(weights, [fm0, fm1])

        Create a mask `tensor` and extract the elements of the `ktensor` using the mask:

        >>> W = ttb.tensor.from_data(np.array([[0, 1], [1, 0]]))
        >>> print(K.mask(W))
        [[63.]
         [39.]]
        """
        # Error check
        if len(W.shape) != len(self.shape) or np.any(
            np.array(W.shape) > np.array(self.shape)
        ):
            assert False, "Mask cannot be bigger than the data tensor"

        # Extract locations of nonzeros in W
        wsubs, _ = W.find()

        # Assemble return array
        nvals = wsubs.shape[0]
        vals = np.zeros((nvals, 1))
        for j in range(self.ncomponents):
            tmpvals = self.weights[j] * np.ones((nvals, 1))
            for k in range(self.ndims):
                akvals = self.factor_matrices[k][wsubs[:, [k]], j]
                tmpvals = tmpvals * akvals
            vals = vals + tmpvals
        return vals

    def mttkrp(self, U, n):
        """
        Matricized `tensor` times Khatri-Rao product for `ktensor`.

        Efficiently calculates the matrix product of the n-mode matricization of the `ktensor` with the
        Khatri-Rao product of all entries in U, a list of matrices, except the nth.

        Parameters
        ----------
        U: list of factor matrices
        n: multiplies by all modes except n

        Returns
        -------
        :class:`numpy.ndarray`

        Examples
        --------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> U = [np.ones((2, 2)), np.ones((3, 2)), np.ones(((4, 2)))]
        >>> print(K.mttkrp(U, 0))
        [[24. 24.]
         [24. 24.]]
        """
        if not isinstance(U, list):
            assert False, "Second argument must be list of numpy.ndarray's"

        if len(U) != self.ndims:
            assert False, "List of factor matrices is the wrong length"

        # Number of columns in input matrices
        if n == 0:
            R = U[1].shape[1]
        else:
            R = U[0].shape[1]

        # Compute matrix of weights
        W = np.tile(self.weights[:, None], (1, R))
        for i in range(self.ndims):
            if i != n:
                W = W * (self.factor_matrices[i].T @ U[i])

        # Find each column of answer by multiplying columns of X.u{n} with weights
        return self.factor_matrices[n] @ W

    @property
    def ncomponents(self):
        """Number of components (i.e., number of columns in each factor matrix) of the `ktensor`.

        Returns
        -------
        :int
        """
        return len(self.weights)

    @property
    def ndims(self):
        """Number of dimensions (i.e., number of factor_matrices) of the `ktensor`.

        Returns
        -------
        :int
        """
        return len(self.factor_matrices)

    def norm(self):
        """
        Compute the norm (i.e., square root of the sum of squares of entries) of a `ktensor`.

        Returns
        --------
        :int

        Example
        -------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> K.norm()
        9.797958971132712
        """
        # Compute the matrix of correlation coefficients
        coefMatrix = self.weights[:, None] @ self.weights[None, :]
        for f in self.factor_matrices:
            coefMatrix = coefMatrix * (f.T @ f)
        return np.sqrt(np.abs(np.sum(coefMatrix)))

    def normalize(self, weight_factor=None, sort=False, normtype=2, mode=None):
        """
        Normalize the columns of the factor matrices.

        Parameters
        ----------
        weight_factor: {"all", int}, optional
            Absorb the weights into one or more factors:
               * "all": absorb weight equally across all factors
               * int: absorb weight into a single dimension (int must be in range(self.ndims)
        sort: bool
            Sort the columns in descending order of the weights
        normtype: {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
            Order of the norm (see :func:`numpy.linalg.norm` for possible values)
        mode: {int, None}
            Index of factor matrix to normalize. A value of `None` means normalize all factor matrices.

        Returns
        --------
        :class:`pyttb.ktensor`

        Example
        -------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> print(K.normalize())  # doctest: +ELLIPSIS
        ktensor of shape 2 x 3 x 4
        weights=[4.898... 4.898...]
        factor_matrices[0] =
        [[0.7071... 0.7071...]
         [0.7071... 0.7071...]]
        factor_matrices[1] =
        [[0.5773... 0.5773...]
         [0.5773... 0.5773...]
         [0.5773... 0.5773...]]
        factor_matrices[2] =
        [[0.5 0.5]
         [0.5 0.5]
         [0.5 0.5]
         [0.5 0.5]]
        """
        # when mode is specified, just normalize self.factor_matrices[mode]
        if mode is not None:
            if mode in range(self.ndims):
                for r in range(self.ncomponents):
                    tmp = np.linalg.norm(self.factor_matrices[mode][:, r], ord=normtype)
                    if tmp > 0:
                        self.factor_matrices[mode][:, r] = (
                            1.0 / tmp * self.factor_matrices[mode][:, r]
                        )
                    self.weights[r] = self.weights[r] * tmp
                return
            else:
                assert (
                    False
                ), "Parameter single_factor is invalid; index must be an int in range of number of dimensions"

        # ensure that all factor_matrices are normalized
        for mode in range(self.ndims):
            for r in range(self.ncomponents):
                tmp = np.linalg.norm(self.factor_matrices[mode][:, r], ord=normtype)
                if tmp > 0:
                    self.factor_matrices[mode][:, r] = (
                        1.0 / tmp * self.factor_matrices[mode][:, r]
                    )
                self.weights[r] = self.weights[r] * tmp

        # check that all weights are positive, flip sign of columns in first factor matrix if negative weight found
        idx = np.where(self.weights < 0)
        self.factor_matrices[0][:, idx] = -self.factor_matrices[0][:, idx]
        self.weights[idx] = -self.weights[idx]

        # absorb weight into factors
        if weight_factor == "all":
            # all factors
            D = np.diag(np.power(self.weights, 1.0 / self.ndims))
            for i in range(self.ndims):
                self.factor_matrices[i] = self.factor_matrices[i] @ D
            self.weights[:] = 1.0
        elif weight_factor in range(self.ndims):
            # single factor
            self.factor_matrices[weight_factor] = self.factor_matrices[
                weight_factor
            ] @ np.diag(self.weights)
            self.weights = np.ones((self.weights.shape))

        if sort:
            if self.ncomponents > 1:
                # indices of srting in descending order
                p = np.argsort(self.weights)[::-1]
                self = self.arrange(permutation=p)

        return self

    def nvecs(self, n, r, flipsign=True):
        """
        Compute the leading mode-n vectors for a `ktensor`.

        Computes the `r` leading eigenvectors of Xn*Xn'
        (where Xn is the mode-N matricization/unfolding of self), which provides
        information about the mode-N fibers. In two-dimensions, the `r`
        leading mode-1 vectors are the same as the 'r' left singular vectors
        and the `r` leading mode-2 vectors are the same as the `r` right
        singular vectors. By default, this method computes the top `r`
        eigenvectors of the matrix Xn*Xn'.

        Parameters
        ----------
        n: int
            Mode for tensor matricization
        r: int
            Number of eigenvectors to compute and use
        flipsign: bool
            Make each column's largest element positive if `True`

        Returns
        -------
        :class:`Numpy.ndarray`

        Example
        -------
        Compute single eigenvector for dimension 0:

        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> nvecs1 = K.nvecs(0, 1)
        >>> print(nvecs1) # doctest: +ELLIPSIS
        [[0.70710678...]
         [0.70710678...]]

        Compute first 2 leading eigenvectors for dimension 0:

        >>> nvecs2 = K.nvecs(0, 2)
        >>> print(nvecs2) # doctest: +ELLIPSIS
        [[ 0.70710678...  0.70710678...]
         [ 0.70710678... -0.70710678...]]
        """
        M = self.weights[:, None] @ self.weights[:, None].T
        for i in range(self.ndims):
            if i != n:
                M = M * (self.factor_matrices[i].T @ self.factor_matrices[i])

        y = self.factor_matrices[n] @ M @ self.factor_matrices[n].T

        if r < y.shape[0] - 1:
            w, v = scipy.sparse.linalg.eigsh(y, r)
            v = v[:, (-np.abs(w)).argsort()]
            v = v[:, :r]
        else:
            warnings.warn(
                "Greater than or equal to ktensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
            )
            w, v = scipy.linalg.eigh(y)
            v = v[:, (-np.abs(w)).argsort()]
            v = v[:, :r]

        if flipsign:
            idx = np.argmax(np.abs(v), axis=0)
            for i in range(v.shape[1]):
                if v[idx[i], i] < 0:
                    v[:, i] *= -1
        return v

    def permute(self, order):
        """
        Permute `ktensor` dimensions.

        Rearranges the dimensions of A so that they are in the order specified by the
        vector `order`. The output is a `ktensor` with components rearranged as specified
        by `order`. The corresponding tensor has the same components as `self` but the order
        of the subscripts needed to access any particular element is rearranged as
        specified by `order`.

        Parameters
        ----------
        order: :class:`Numpy.ndarray`

        Returns
        -------
        :class:`pyttb.ktensor`
            shapeNew == shapePrevious[order]

        Example
        -------
        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor.from_data(weights, factor_matrices)
        >>> print(K)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Permute the order of the dimension so they are in reverse order:

        >>> K1 = K.permute(np.array([1, 0]))
        >>> print(K1)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[5. 6.]
         [7. 8.]]
        factor_matrices[1] =
        [[1. 2.]
         [3. 4.]]
        """
        # Check that the permutation is valid
        if tuple(range(self.ndims)) != tuple(sorted(order.tolist())):
            assert False, "Invalid permutation"

        return ktensor.from_data(self.weights, [self.factor_matrices[i] for i in order])

    def redistribute(self, mode):
        """
        Distribute weights of a `ktensor` to a specified mode

        Parameters
        ----------
        mode: int

        Example
        -------
        Create a `ktensor`:

        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor.from_data(weights, factor_matrices)
        >>> print(K)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Distribute weights of that `ktensor` to mode 0:

        >>> K.redistribute(0)
        >>> print(K)
        ktensor of shape 2 x 2
        weights=[1. 1.]
        factor_matrices[0] =
        [[1. 4.]
         [3. 8.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]
        """
        for r in range(self.ncomponents):
            self.factor_matrices[mode][:, [r]] = (
                self.factor_matrices[mode][:, [r]] * self.weights[r]
            )
            self.weights[r] = 1

    @property
    def shape(self):
        """Shape of a `ktensor`.

        Returns the sizes of all dimensions.

        Returns
        -------
        :class:`tuple`
        """
        return tuple([f.shape[0] for f in self.factor_matrices])

    def score(self, other, weight_penalty=True, threshold=0.99, greedy=True):
        """
        Checks if two ktensor instances match except for permutation.

        We define matching as follows. If A (self) and B (other) are single component
        ktensors that have been normalized so that their weights are weights_a and
        weights_b, then the score is defined as

            score = penalty * (a1.T*b1) * (a2.T*b2) * ... * (aR.T*bR),

        where the penalty is defined by the weights such that

            penalty = 1 - abs(weights_a - weights_b) / max(weights_a, weights_b).

        The score of multi-component ktensors is a normalized sum of the
        scores across the best permutation of the components of A. A can have
        more components than B --- any extra components are ignored in terms of
        the matching score.

        Parameters
        ----------
        other: :class:`pyttb.ktensor`
            `ktensor` to match against
        weight_penalty: bool
            Flag indicating whether or not to consider the weights in the calculations.
            Default: true
        threshold: float
            Threshold specified in the formula above for determining a match.
            Default: 0.99
        greedy: bool
            Flag indicating whether or not to consider all possible matchings
            (exponentially expensive) or just do a greedy matching. Default: true

        Returns
        -------
        int
            Score (between 0 and 1)
        :class:`pyttb.ktensor`
            Copy of `self`, which has been normalized and permuted to best match `other`
        bool
            Flag indicating a match according to a user-specified threshold
        :class:`Numpy.ndarray`
            Permutation (i.e. array of indices of the modes of self) of the components
            of self that was used to best match other

        Example
        -------
        Create two `ktensor` instances:

        >>> A = ttb.ktensor.from_data(np.array([2., 1., 3.]), np.ones((3,3)), np.ones((4,3)), np.ones((5,3)))
        >>> B = ttb.ktensor.from_data(np.array([2., 4.]), np.ones((3,2)), np.ones((4,2)), np.ones((5,2)))
        >>> score,Aperm,flag,perm = A.score(B)  #  Compute `score` using `ktensor.weights`
        >>> print(score)
        0.875
        >>> print(perm)
        [0 2 1]

        Compute `score` not using `ktensor.weights`:

        >>> score,Aperm,flag,perm = A.score(B,weight_penalty=False)
        >>> print(score)
        1.0
        >>> print(perm)
        [0 1 2]
        """

        if not greedy:
            assert (
                False
            ), "Not yet implemented. Only greedy method is implemented currently."

        if not isinstance(other, ktensor):
            assert False, "The first input should be a ktensor"

        if not (self.shape == other.shape):
            assert False, "Size mismatch"

        # Set-up
        N = self.ndims
        RA = self.ncomponents
        RB = other.ncomponents

        # We're matching components in A to B
        if RA < RB:
            assert False, "Tensor A must have at least as many components as tensor B"

        # Make sure columns of factor matrices are normalized
        A = ttb.ktensor.from_tensor_type(self).normalize()
        B = ttb.ktensor.from_tensor_type(other).normalize()

        # Compute all possible vector-vector congruences.

        # Compute every pair for each mode
        Cbig = ttb.tensor.from_function(np.zeros, (RA, RB, N))
        for n in range(N):
            Cbig[:, :, n] = np.abs(A.factor_matrices[n].T @ B.factor_matrices[n])

        # Collapse across all modes using the product
        C = Cbig.collapse(np.array([2]), np.prod).double()

        # Calculate penalty based on differences in the Lambda's
        # Note that we are assuming the the lambda value are positive because the
        # ktensor's were previously normalized.
        if weight_penalty:
            P = np.zeros((RA, RB))
            for ra in range(RA):
                la = A.weights[ra]
                for rb in range(RB):
                    lb = B.weights[rb]
                    if (la == 0) and (lb == 0):
                        # if both lambda values are zero (0), they match
                        P[ra, rb] = 1
                    else:
                        P[ra, rb] = 1 - (
                            np.abs(la - lb) / np.max([np.abs(la), np.abs(lb)])
                        )
            C = P * C

        # Option to do greedy matching
        if greedy:
            best_perm = -1 * np.ones((RA), dtype=int)
            best_score = 0
            for r in range(RB):
                idx = np.argmax(C.reshape(np.prod(C.shape), order="F"))
                ij = tt_ind2sub((RA, RB), idx)
                best_score = best_score + C[ij[0], ij[1]]
                C[ij[0], :] = -10
                C[:, ij[1]] = -10
                best_perm[ij[1]] = ij[0]
            best_score = best_score / RB
            flag = 1

            # Rearrange the components of A according to the best matching
            foo = np.arange(RA)
            tf = np.in1d(foo, best_perm)
            best_perm[RB : RA + 1] = foo[~tf]
            A.arrange(permutation=best_perm)
            return best_score, A, flag, best_perm

    def symmetrize(self):
        """
        Symmetrize a `ktensor` in all modes.

        Symmetrize a `ktensor` with respect to all modes so that the resulting `ktensor` is symmetric with respect
        to any permutation of indices.

        Returns
        -------
        :class:`pyttb.ktensor`
            A new `ktensor` whose factor matrices are symmetric

        Example
        -------
        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor.from_data(weights, factor_matrices)
        >>> print(K)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Make the factor matrices of the `ktensor` symmetric:

        >>> K1 = K.symmetrize()
        >>> print(K1) # doctest: +ELLIPSIS
        ktensor of shape 2 x 2
        weights=[1. 1.]
        factor_matrices[0] =
        [[2.3404... 4.9519...]
         [4.5960... 8.0124...]]
        factor_matrices[1] =
        [[2.3404... 4.9519...]
         [4.5960... 8.0124...]]
        """
        # Check tensor dimensions for compatibility with symmetrization
        assert (
            self.shape == self.shape[0] * np.ones(self.ndims)
        ).all(), "Tensor is not cubic -- cannot be symmetrized"

        # Distribute lambda evenly into factors
        K = self.from_tensor_type(self)
        K.normalize("all")

        weights = K.weights
        factor_matrices = K.factor_matrices
        fm0 = factor_matrices[0]

        V = fm0
        for i in range(1, K.ndims):
            fmi = factor_matrices[i]
            for j in range(fm0.shape[1]):
                if fm0[:, [j]].T @ fmi[:, [j]] < 0:
                    fmi[:, [j]] = -fmi[:, [j]]
                    weights[j] = -weights[j]
            V = V + fmi
        V = V / K.ndims

        # Odd-ordered tensors should not have any negative weights
        if np.mod(K.ndims, 2) == 1:
            for j in range(K.ncomponents):
                if weights[j] < 0:
                    weights[j] = -weights[j]
                    V[:, [j]] = -V[:, [j]]

        return self.from_data(weights, [V.copy() for i in range(K.ndims)])

    def tolist(self, mode=None):
        """
        Converts `ktensor` to a list of factor matrices, evenly distributing the weights across factors.

        Parameters
        ----------
        mode: int
            Index of factor matrix to absorb all of the weight.

        Returns
        -------
        :list of :class:`numpy.ndarray`
            List of factor matrices

        Example
        -------
        Create a `ktensor` of all ones:

        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor.from_data(weights, factor_matrices)
        >>> print(K)
        ktensor of shape 2 x 2
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Spread weights equally to all factors and return list of factor matrices:

        >>> fm_list = K.tolist()
        >>> for fm in fm_list: print(fm) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[1. 2.8284...]
         [3. 5.6568...]]
        [[ 5. 8.4852...]
         [ 7. 11.313...]]

        Shift weight to single factor matrix and return list of factor matrices:

        >>> fm_list = K.tolist(0)
        >>> for fm in fm_list: print(fm)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[ 8.6023... 40. ]
         [25.8069... 80. ]]
        [[0.5812... 0.6...]
         [0.8137... 0.8...]]
        """
        if mode is not None:
            if isinstance(mode, int) and mode in range(self.ndims):
                self.normalize(mode)
                return self.factor_matrices.copy()
            else:
                assert False, "Input parameter'mode' must be in the range of self.ndims"

        # all weights are equal to 1
        if (self.weights == np.ones(self.weights.shape)).all():
            return self.factor_matrices.copy()

        lsgn = np.sign(self.weights)
        D = np.diag(np.power(np.fabs(self.weights), 1.0 / self.ndims))
        factor_matrices = self.factor_matrices.copy()
        factor_matrices[0] = factor_matrices[0] @ np.diag(lsgn)
        for n in range(self.ndims):
            factor_matrices[n] = factor_matrices[n] @ D
        return factor_matrices

    def tovec(self, include_weights=True):
        """
        Convert `ktensor` to column vector.

        Parameters
        ----------
        include_weights: bool
          Flag to specify whether or not to include weights in output (default is True).

        Returns
        -------
        :class:`numpy.ndarray`
          The length of the column vector is (sum(K.shape)+1)*K.ncomponents. The vector contains the ktensor
          weights (if requested) stacked on top of each of the columns of the factor_matrices in order.

        Example
        -------
        Create a `ktensor` from a vector:

        >>> rank = 2
        >>> shape = np.array([2, 3, 4])
        >>> data = np.arange(1, rank*sum(shape)+1)
        >>> weights = 2 * np.ones(rank)
        >>> weights_and_data = np.concatenate((weights, data), axis=0)
        >>> K_from_vector = ttb.ktensor.from_vector(weights_and_data[:], shape, True)
        >>> print(K_from_vector)
        ktensor of shape 2 x 3 x 4
        weights=[2. 2.]
        factor_matrices[0] =
        [[1. 3.]
         [2. 4.]]
        factor_matrices[1] =
        [[ 5.  8.]
         [ 6.  9.]
         [ 7. 10.]]
        factor_matrices[2] =
        [[11. 15.]
         [12. 16.]
         [13. 17.]
         [14. 18.]]

        Create a ktensor from a vector of data extracted from another `ktensor`:

        >>> K_from_tovec = ttb.ktensor.from_vector(K_from_vector.tovec(), shape, True)
        >>> print(K_from_tovec)
        ktensor of shape 2 x 3 x 4
        weights=[2. 2.]
        factor_matrices[0] =
        [[1. 3.]
         [2. 4.]]
        factor_matrices[1] =
        [[ 5.  8.]
         [ 6.  9.]
         [ 7. 10.]]
        factor_matrices[2] =
        [[11. 15.]
         [12. 16.]
         [13. 17.]
         [14. 18.]]
        """
        if include_weights:
            x = np.zeros(self.ncomponents * int(sum(self.shape) + 1))
            x[0 : self.ncomponents] = self.weights
            offset = self.ncomponents
        else:
            x = np.zeros(self.ncomponents * int(sum(self.shape)))
            offset = 0

        for f in self.factor_matrices:
            for r in range(self.ncomponents):
                x[offset : offset + f.shape[0]] = f[:, r].reshape(f.shape[0])
                offset += f.shape[0]

        # if include_weights:
        #     x = np.zeros((self.ncomponents * int(sum(self.shape)+1), 1))
        #     x[0:self.ncomponents] = self.weights.reshape((len(self.weights), 1))
        #     offset = self.ncomponents
        # else:
        #     x = np.zeros((self.ncomponents * int(sum(self.shape)), 1))
        #     offset = 0
        #
        # for f in self.factor_matrices:
        #     for r in range(self.ncomponents):
        #         x[offset:offset+f.shape[0]] = f[:, r].reshape((f.shape[0], 1))
        #         offset += f.shape[0]
        return x

    def ttv(self, vector, dims=None):
        """
        `Tensor` times vector for `ktensors`.

        Computes the product of a `ktensor` with a vector.  If `dims` is an integer, it specifies
        the dimension in the `ktensor` along which the vector is multiplied.  If the shape of the vector
        is = (I,1), then the size of dimension `dims` of the `ktensor` must have size I.  Note that
        number of dimensions of the returned `tensor` is 1 less than the dimension of the `ktensor` used
        in the multiplication because dimension `dims` is removed.

        ttv(np.array([v1,v2,...,vN])) computes the product of the `ktensor` with a sequence of vectors in the
        array.  The products are computed sequentially along all dimensions (or modes) of the `ktensor`. The array
        contains N=self.ndims vectors.

        ttv(np.array([v1,v2,...,vk]), dims=np.array([I1,I2,...,Ik) computes the sequence of `ktensor`-by-vector
        products along the dimensions specified by `dims`. In this case, the number of products, k, can be less
        than N=self.ndims and the order of the sequence does not need to match the order of the dimensions in the
        `ktensor`. Note that the number of vectors must match the number of dimensions provides, and the length of
        each vector must match the size of each dimension of the `ktensor` specified in `dims`.

        Parameters
        ----------
        vector: :class:`Numpy.ndarray`, list[:class:`Numpy.ndarray`]
        dims: int, :class:`Numpy.ndarray`

        Returns
        -------
        float or :class:`pyttb.ktensor`
            The number of dimensions of the returned `ktensor` is N-k, where N=self.ndims and k = number of
            vectors provided as input. If k == N, a scalar is returned

        Example
        -------
        Compute the product of a `ktensor` and a single vector (results in a `ktensor`):

        >>> rank = 2
        >>> shape = np.array([2, 3, 4])
        >>> data = np.arange(1, rank*sum(shape)+1)
        >>> weights = 2 * np.ones(rank)
        >>> weights_and_data = np.concatenate((weights, data), axis=0)
        >>> K = ttb.ktensor.from_vector(weights_and_data[:], shape, True)
        >>> K0 = K.ttv(np.array([1, 1, 1]), dims=1) # compute along a single dimension
        >>> print(K0)
        ktensor of shape 2 x 4
        weights=[36. 54.]
        factor_matrices[0] =
        [[1. 3.]
         [2. 4.]]
        factor_matrices[1] =
        [[11. 15.]
         [12. 16.]
         [13. 17.]
         [14. 18.]]

        Compute the product of a `ktensor` and a vector for each dimension (results in a `float`):

        >>> vec2 = np.array([1, 1])
        >>> vec3 = np.array([1, 1, 1])
        >>> vec4 = np.array([1, 1, 1, 1])
        >>> K1 = K.ttv([vec2, vec3, vec4])
        >>> print(K1)
        30348.0

        Compute the product of a `ktensor` and multiple vectors out of order (results in a `ktensor`):

        >>> K2 = K.ttv([vec4, vec3], np.array([2, 1]))
        >>> print(K2)
        ktensor of shape 2
        weights=[1800. 3564.]
        factor_matrices[0] =
        [[1. 3.]
         [2. 4.]]
        """

        if dims is None:
            dims = np.array([])
        elif isinstance(dims, (float, int)):
            dims = np.array([dims])

        # Check that vector is a list of vectors, if not place single vector as element in list
        if len(vector) > 0 and isinstance(vector[0], (int, float, np.int_, np.float_)):
            return self.ttv([vector], dims)

        # Get sorted dims and index for multiplicands
        dims, vidx = ttb.tt_dimscheck(dims, self.ndims, len(vector))

        # Check that each multiplicand is the right size.
        for i in range(dims.size):
            if vector[vidx[i]].shape != (self.shape[dims[i]],):
                assert False, "Multiplicand is wrong size"

        # Figure out which dimensions will be left when we're done
        remdims = np.setdiff1d(range(self.ndims), dims)

        # Collapse dimensions that are being multiplied out
        new_weights = self.weights.copy()
        for i in range(len(dims)):
            new_weights = new_weights * (
                self.factor_matrices[dims[i]].T @ vector[vidx[i]]
            )

        # Create final result
        if len(remdims) == 0:
            return sum(new_weights)
        else:
            factor_matrices = []
            for i in remdims:
                factor_matrices.append(self.factor_matrices[i])
            return ktensor.from_data(new_weights, factor_matrices)

    def update(self, modes, data):
        """
        Updates a `ktensor` in the specific dimensions, `modes`, with the values in `data`
        (in vector or matrix form). The value of `mode` must be an integer between 1
        and self.ndims. Further, the number of elements in `data` must equal self.shape[mode] * self.ncomponents.

        Parameters
        ----------
        modes: list
            List of dimensions to update; values must be in ascending order. Can include one or more of the following values:
              * -1: update the weights
              * `int`: value must be sorted and in `{range(self.ndims)}`
        data: :class:numpy.ndarray
            Vector of data values to use in the update

        Results
        -------
        :class:`pyttb.ktensor`
            The update is performed in place and a reference to self is returned

        Example
        -------
        Create a `ktensor` of all ones:

        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)

        Create vectors for updating various factor matrices of the `ktensor`:

        >>> vec0 = 2 * np.ones(K.shape[0] * K.ncomponents)
        >>> vec1 = 3 * np.ones(K.shape[1] * K.ncomponents)
        >>> vec2 = 4 * np.ones(K.shape[2] * K.ncomponents)

        Update a single factor matrix:

        >>> K1 = K.copy()
        >>> K1 = K1.update(0, vec0)
        >>> print(K1)
        ktensor of shape 2 x 3 x 4
        weights=[1. 1.]
        factor_matrices[0] =
        [[2. 2.]
         [2. 2.]]
        factor_matrices[1] =
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
        factor_matrices[2] =
        [[1. 1.]
         [1. 1.]
         [1. 1.]
         [1. 1.]]

        Update all factor matrices:

        >>> K2 = K.copy()
        >>> vec_all = np.concatenate((vec0, vec1, vec2))
        >>> K2 = K2.update([0, 1, 2], vec_all)
        >>> print(K2)
        ktensor of shape 2 x 3 x 4
        weights=[1. 1.]
        factor_matrices[0] =
        [[2. 2.]
         [2. 2.]]
        factor_matrices[1] =
        [[3. 3.]
         [3. 3.]
         [3. 3.]]
        factor_matrices[2] =
        [[4. 4.]
         [4. 4.]
         [4. 4.]
         [4. 4.]]

        Update some but not all factor matrices

        >>> K3 = K.copy()
        >>> vec_some = np.concatenate((vec0, vec2))
        >>> K3 = K3.update([0, 2], vec_some)
        >>> print(K3)
        ktensor of shape 2 x 3 x 4
        weights=[1. 1.]
        factor_matrices[0] =
        [[2. 2.]
         [2. 2.]]
        factor_matrices[1] =
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
        factor_matrices[2] =
        [[4. 4.]
         [4. 4.]
         [4. 4.]
         [4. 4.]]

        """
        if not isinstance(modes, int):
            assert modes == sorted(modes), "Modes must be sorted in ascending order"
        else:
            modes = [modes]

        loc = 0  # Location in data array
        for k in modes:
            if k == -1:
                # update weights
                endloc = loc + self.ncomponents
                if len(data) < endloc:
                    assert False, "Data is too short"
                self.weights = data[loc:endloc].copy()
                loc = endloc
            elif k < self.ndims:
                # update factor matrix
                endloc = loc + self.shape[k] * self.ncomponents
                if len(data) < endloc:
                    assert False, "Data is too short"
                self.factor_matrices[k] = np.reshape(
                    data[loc:endloc].copy(), (self.shape[k], self.ncomponents)
                )
                loc = endloc
            else:
                assert False, "Invalid mode: {}".format(k)

        ## Check that we used all the data
        if not (loc == len(data)):
            warnings.warn("Failed to consume all of the input data")

        return self

    def __add__(self, other):
        """
        Binary addition for `ktensors`.

        Parameters
        ----------
        other: :class:`pyttb.ktensor`

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        # TODO include test of other as sumtensor and call sumtensor.__add__
        if not isinstance(other, ktensor):
            assert False, "Cannot add instance of this type to a ktensor"

        if self.shape != other.shape:
            assert False, "Must be two ktensors of the same shape"

        weights = np.concatenate((self.weights, other.weights))
        factor_matrices = []
        for k in range(self.ndims):
            factor_matrices.append(
                np.concatenate(
                    (self.factor_matrices[k], other.factor_matrices[k]), axis=1
                )
            )
        return ktensor.from_data(weights, factor_matrices)

    def __getitem__(self, item):
        """
        Subscripted reference for a `ktensor`.

        Subscripted reference is used to query the components of a `ktensor`.

        Parameters
        ----------
        item: {tuple(int), int}

        Example
        -------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> K.weights #<--returns the weights array (np.array([1., 1.]))
        array([1., 1.])
        >>> K.factor_matrices #<--returns a list of 3 np.ndarrays
        [array([[1., 1.],
               [1., 1.]]), array([[1., 1.],
               [1., 1.],
               [1., 1.]]), array([[1., 1.],
               [1., 1.],
               [1., 1.],
               [1., 1.]])]
        >>> K.factor_matrices[0] #<--returns the factor matrix corresponding to the first mode
        array([[1., 1.],
               [1., 1.]])
        >>> K[0] #<-returns the factor matrix corresponding to the first mode
        array([[1., 1.],
               [1., 1.]])
        >>> K[1, 2, 0] #<--calculates and returns a specific single element of K
        2.0
        >>> K[0][:, [0]] # returns the first column of the factor matrix corresponding to the first mode
        array([[1.],
               [1.]])
        """
        if isinstance(item, tuple):
            if len(item) == self.ndims:
                # Extract single element
                a = 0
                for k in range(self.ncomponents):
                    b = self.weights[k]
                    for i in range(self.ndims):
                        b = b * self.factor_matrices[i][item[i], k]
                    a = a + b
                return a
            else:
                assert (
                    False
                ), "ktensor.__getitem__ requires tuples with {} elements".format(
                    self.ndims
                )
        elif isinstance(item, (int, np.int_)) and item in range(self.ndims):
            # Extract factor matrix
            return self.factor_matrices[item].copy()
        else:
            assert (
                False
            ), "ktensor.__getitem__() can only extract single elements (tuple of indices) or factor matrices (single index)"

    def __neg__(self):
        """
        Unary minus (negative) for ktensors.

        Parameters
        ----------
        other: :class:`pyttb.ktensor`

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        return ktensor.from_data(-self.weights, self.factor_matrices)

    def __pos__(self):
        """
        Unary minus (positive) for `ktensors`.

        Parameters
        ----------
        other: :class:`pyttb.ktensor`

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        return ktensor.from_tensor_type(self)

    def __setitem__(self, key, value):
        """
        Subscripted assignment for `ktensors`.

        Subscripted assignment cannot be used to update individual elements of a `ktensor`. However, you
        can update the weights vector or the factor matrices of a `ktensor`. The entire factor matrix or weight

        Example
        -------
        >>> K = ktensor.from_data(np.ones((4,1)), [np.random.random((2,4)), np.random.random((3,4)), np.random.random((4,4))])
        >>> K.weights = 2 * np.ones((4,1))
        >>> K.factor_matrices[0] = np.zeros((2, 4))
        >>> K.factor_matrices = [np.zeros((2, 4)), np.zeros((3, 4)), np.zeros((4, 4))]
        """
        assert (
            False
        ), "Subscripted assignment cannot be used to update individual elements of a ktensor. However, \
        you can update the weights vector or the factor matrices of a ktensor. The entire factor matrix or weight \
        vector must be provided."

    def __sub__(self, other):
        """
        Binary subtraction for ktensors.

        Parameters
        ----------
        other: :class:`pyttb.ktensor`

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        if not isinstance(other, ktensor):
            assert False, "Cannot subtract instance of this type from a ktensor"

        if self.shape != other.shape:
            assert False, "Must be two ktensors of the same shape"

        weights = np.concatenate((self.weights, -other.weights))
        factor_matrices = []
        for k in range(self.ndims):
            factor_matrices.append(
                np.concatenate(
                    (self.factor_matrices[k], other.factor_matrices[k]), axis=1
                )
            )
        return ktensor.from_data(weights, factor_matrices)

    def __mul__(self, other):
        """
        Elementwise (including scalar) multiplication for ktensors.

        Parameters
        ----------
        other: :class:`pyttb.tensor`, :class:`pyttb.sptensor`, float, int

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        if isinstance(other, (ttb.sptensor, ttb.tensor)):
            return other.__mul__(self)

        if isinstance(other, (float, int)):
            return ktensor.from_data(other * self.weights, self.factor_matrices)

        assert (
            False
        ), "Multiplication by ktensors only allowed for scalars, tensors, or sptensors"

    def __rmul__(self, other):
        """
        Elementwise (including scalar) multiplication for ktensors.

        Parameters
        ----------
        other: :class:`pyttb.tensor`, :class:`pyttb.sptensor`, float, int

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        return self.__mul__(other)

    def __repr__(self):
        """
        String representation of a ktensor.

        Returns
        -------
        str:
          Contains the shape, weights and factor_matrices as strings on different lines.
        """
        s = ""
        s += "ktensor of shape "
        s += (" x ").join([str(int(d)) for d in self.shape])
        s += "\n"
        s += "weights="
        s += str(self.weights)
        for i in range(len(self.factor_matrices)):
            s += "\nfactor_matrices[{}] =\n".format(i)
            s += str(self.factor_matrices[i])
        return s

    __str__ = __repr__


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
