"""Classes and functions for working with Kruskal tensors."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from math import prod
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
    overload,
)

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg

import pyttb as ttb
from pyttb.pyttb_utils import (
    OneDArray,
    Shape,
    get_mttkrp_factors,
    isrow,
    isvector,
    np_to_python,
    parse_one_d,
    parse_shape,
    to_memory_order,
    tt_dimscheck,
    tt_ind2sub,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class ktensor:
    """
    KTENSOR Class for Kruskal tensors (decomposed).

    Contains the following data members:

    ``weights``: :class:`numpy.ndarray` vector containing the weights of the
    rank-1 tensors defined by the outer products of the column vectors of the
    factor_matrices.

    ``factor_matrices``: :class:`list` of :class:`numpy.ndarray`. The length
    of the list is equal to the number of dimensions of the tensor. The shape
    of the ith element of the list is (n_i, r), where n_i is the length
    dimension i and r is the rank of the tensor (as well as the length of the
    weights vector).

    Instances of :class:`pyttb.ktensor` can be created using `__init__()` or
    one of the following methods:

      * :meth:`from_function`
      * :meth:`from_vector`

    Examples
    --------
    For all examples listed below, the following module imports are assumed:

    >>> import pyttb as ttb
    >>> import numpy as np
    """

    __slots__ = ("factor_matrices", "weights")

    def __init__(  # noqa: PLR0912
        self,
        factor_matrices: Sequence[np.ndarray] | None = None,
        weights: np.ndarray | None = None,
        copy: bool = True,
    ):
        """Create a :class:`pyttb.ktensor`.

        Created in one of the following ways:
          - With no inputs (or `weights` and `factor_matrices` both None),
            return an empty :class:`pyttb.ktensor`.
          - Otherwise, return a :class:`pyttb.ktensor` with `weights` and
            `factor_matrices` as provided.

        Parameters
        ----------
        factor_matrices:
            Factors for ktensor.
        weights:
            Tensor weights, defaults to all 1's.
        copy:
            Whether or not to copy the input data or just reference it.

        Examples
        --------
        Create an empty :class:`pyttb.ktensor`:

        >>> K = ttb.ktensor()
        >>> print(K)
        ktensor of shape () with order F
        weights=[]
        factor_matrices=[]

        Create a :class:`pyttb.ktensor` from weights and a list of factor
        matrices:

        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> K = ttb.ktensor([fm0, fm1], weights)
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Create a :class:`pyttb.ktensor` from a :class:`list` of factor
        matrices (without providing weights):

        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor([fm0, fm1])
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 1.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]
        """
        # Cannot specify weights and not factor_matrices
        if factor_matrices is None and weights is not None:
            assert False, "factor_matrices cannot be None if weights are provided."

        # Empty constructor
        if factor_matrices is None and weights is None:
            self.weights = np.array([], order=self.order)
            self.factor_matrices: list[np.ndarray] = []
            return

        # 'factor_matrices' must be a list
        if not isinstance(factor_matrices, Sequence):
            assert False, "Input 'factor_matrices' must be a sequence."
        # each factor matrix should be a np.ndarray
        if not (
            all(isinstance(fm, np.ndarray) for fm in factor_matrices)
            and all(fm.dtype == float for fm in factor_matrices)
        ):
            assert False, (
                "Each item in 'factor_matrices' must be a numpy.ndarray object with "
                "dtype=float."
            )
        # the number of columns of all factor_matrices must be equal
        num_components = factor_matrices[0].shape[1]
        if not all(fm.shape[1] == num_components for fm in factor_matrices):
            assert False, (
                "The number of columns each item in 'factor_matrices' must be the same."
            )

        # process weights
        if weights is not None:
            # check if weights are the correct type and shape
            assert (
                isinstance(weights, np.ndarray)
                and weights.dtype == float
                and weights.shape == (num_components,)
            ), (
                "Input 'weights' must be a numpy.ndarray object with dtype=float and "
                "length equal to the number of columns in each factor matrix."
            )
            # make copy or use reference
            if copy:
                self.weights = weights.copy(self.order)
            else:
                if not self._matches_order(weights):
                    logging.warning(
                        f"Selected no copy, but input weights aren't {self.order} "
                        "ordered so must copy."
                    )
                self.weights = to_memory_order(weights, self.order)
        else:
            # create weights if not provided
            self.weights = np.ones(num_components, order=self.order)

        # process factor_matrices
        if copy:
            self.factor_matrices = [fm.copy(order=self.order) for fm in factor_matrices]
        else:
            if not all(self._matches_order(factor) for factor in factor_matrices):
                logging.warning(
                    "Selected no copy, but input factor matrices aren't "
                    f"{self.order} ordered so must copy."
                )
                factor_matrices = [
                    to_memory_order(fm, self.order, copy=True) for fm in factor_matrices
                ]
            if not isinstance(factor_matrices, list):
                logging.warning("Must provide factor matrices as list to avoid copy")
                factor_matrices = list(factor_matrices)
            self.factor_matrices = factor_matrices

    @classmethod
    def from_function(
        cls,
        function_handle: Callable[[tuple[int, ...]], np.ndarray],
        shape: Shape,
        num_components: int,
    ):
        """Construct a :class:`pyttb.ktensor`.

        Factor matrix entries are
        set using a function. The weights of the returned
        :class:`pyttb.ktensor` will all be equal to 1.

        Parameters
        ----------
        function_handle:
            A function that can accept a shape (i.e., :class:`tuple` of
            dimension sizes) and return a :class:`numpy.ndarray` of that shape.
            Example functions include `numpy.random.random_sample`,
            `numpy.zeros`, `numpy.ones`.
        shape:
            Shape of the resulting tensor.
        num_components:
            Number of components/weights for resulting tensor.

        Returns
        -------
        Constructed ktensor.

        Examples
        --------
        Create a :class:`pyttb.ktensor` with entries of the factor matrices
        taken from a uniform random distribution:

        >>> np.random.seed(1)
        >>> K = ttb.ktensor.from_function(np.random.random_sample, (2, 3, 4), 2)
        >>> print(K)  # doctest: +ELLIPSIS
        ktensor of shape (2, 3, 4) with order F
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

        Create a :class:`pyttb.ktensor` with entries equal to 1:

        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> print(K)
        ktensor of shape (2, 3, 4) with order F
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

        Create a :class:`pyttb.ktensor` with entries equal to 0:

        >>> K = ttb.ktensor.from_function(np.zeros, (2, 3, 4), 2)
        >>> print(K)
        ktensor of shape (2, 3, 4) with order F
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
        assert callable(function_handle), "Input parameter 'fun' must be a function."
        shape = parse_shape(shape)
        assert isinstance(num_components, int), (
            "Input parameter 'num_components' must be an int."
        )
        nd = len(shape)
        weights = np.ones(num_components)
        factor_matrices = []
        for i in range(nd):
            factor_matrices.append(function_handle((shape[i], num_components)))
        return cls(factor_matrices, weights, copy=False)

    @classmethod
    def from_vector(cls, data: np.ndarray, shape: Shape, contains_weights: bool):
        """Construct a :class:`pyttb.ktensor` from a vector and shape.

        The rank of the
        :class:`pyttb.ktensor` is inferred from the shape and length of the vector.

        Parameters
        ----------
        data:
            Vector containing either elements of the factor matrices or elements of the
            weights and factor matrices. When both the elements of
            the weights and the factor_matrices are present, the weights come
            first and the columns of the factor matrices come next.
        shape:
            Shape of the resulting ktensor.
        contains_weights:
            Flag to specify if `data` contains weights.
            If False, all weights are set to 1.

        Returns
        -------
        Constructed ktensor.

        Examples
        --------
        Create a :class:`pyttb.ktensor` from a vector containing only
        elements of the factor matrices:

        >>> rank = 2
        >>> shape = (2, 3, 4)
        >>> data = np.arange(1, rank * sum(shape) + 1).astype(float)
        >>> K = ttb.ktensor.from_vector(data[:], shape, False)
        >>> print(K)
        ktensor of shape (2, 3, 4) with order F
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

        Create a :class:`pyttb.ktensor` from a vector containing elements
        of both the weights and the factor matrices:

        >>> weights = 2 * np.ones(rank).astype(float)
        >>> weights_and_data = np.concatenate((weights, data), axis=0)
        >>> K = ttb.ktensor.from_vector(weights_and_data[:], shape, True)
        >>> print(K)
        ktensor of shape (2, 3, 4) with order F
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
        shape = parse_shape(shape)
        assert isinstance(contains_weights, bool), (
            "Input parameter 'contains_weights' must be a bool."
        )

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
            weights = data[0:num_components].copy("K")
            shift = num_components
        else:
            weights = np.ones(num_components)
            shift = 0

        # extract factor matrices
        factor_matrices = []
        for n, shape_n in enumerate(shape):
            mstart = num_components * sum(shape[0:n]) + shift
            mend = num_components * sum(shape[0 : n + 1]) + shift
            # the following will match MATLAB output
            factor_matrix = np.reshape(
                data[mstart:mend].copy("K"), (shape_n, num_components), order="F"
            )
            factor_matrices.append(factor_matrix)

        return cls(factor_matrices, weights, copy=False)

    @property
    def order(self) -> Literal["F"]:
        """Return the data layout of the underlying storage."""
        return "F"

    def _matches_order(self, array: np.ndarray) -> bool:
        """Check if provided array matches tensor memory layout."""
        if array.flags["C_CONTIGUOUS"] and self.order == "C":
            return True
        if array.flags["F_CONTIGUOUS"] and self.order == "F":
            return True
        return False

    def arrange(
        self,
        weight_factor: int | None = None,
        permutation: tuple | list | np.ndarray | None = None,
    ):
        """Arrange the rank-1 components of a :class:`pyttb.ktensor` in place.

        If `permutation` is passed, the columns of `self.factor_matrices` are
        arranged using the provided permutation, so you must make a copy
        before calling this method if you want to store the original
        :class:`pyttb.ktensor`. If `weight_factor` is passed, then the values
        in `self.weights` are absorbed into
        `self.factor_matrices[weight_factor]`. If no parameters are passed,
        then the columns of `self.factor_matrices` are normalized and then
        permuted such that the resulting `self.weights` are sorted by
        magnitude, greatest to least. Passing both parameters leads to an
        error.

        Parameters
        ----------
        weight_factor:
            Index of the factor matrix the weights will be absorbed into.
        permutation:
            The new order of the components of the :class:`pyttb.ktensor`
            into which to permute. The permutation must be of length equal to
            the number of components of the :class:`pyttb.ktensor`, `self.ncomponents`
            and must be a permutation of [0,...,`self.ncomponents`-1].

        Examples
        --------
        Create the initial :class:`pyttb.ktensor`:

        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> K = ttb.ktensor([fm0, fm1], weights)
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Arrange the columns of the factor matrices using a permutation:

        >>> p = [1, 0]
        >>> K.arrange(permutation=p)
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[2. 1.]
        factor_matrices[0] =
        [[2. 1.]
         [4. 3.]]
        factor_matrices[1] =
        [[6. 5.]
         [8. 7.]]

        Normalize and permute columns such that `weights` are sorted in
        decreasing order:

        >>> K.arrange()
        >>> print(K)  # doctest: +ELLIPSIS
        ktensor of shape (2, 2) with order F
        weights=[89.4427... 27.2029...]
        factor_matrices[0] =
        [[0.4472... 0.3162...]
         [0.8944... 0.9486...]]
        factor_matrices[1] =
        [[0.6... 0.5812...]
         [0.8... 0.8137...]]

        Absorb the weights into the second factor:

        >>> K.arrange(weight_factor=1)
        >>> print(K)  # doctest: +ELLIPSIS
        ktensor of shape (2, 2) with order F
        weights=[1. 1.]
        factor_matrices[0] =
        [[0.4472... 0.3162...]
         [0.8944... 0.9486...]]
        factor_matrices[1] =
        [[53.6656... 15.8113...]
         [71.5541... 22.1359...]]
        """
        if permutation is not None and weight_factor is not None:
            assert False, (
                "Weighting and permuting the ktensor at the same time is not allowed."
            )

        # arrange columns of factor matrices using the permutation provided
        if permutation is not None and isinstance(
            permutation, (tuple, list, np.ndarray)
        ):
            if len(permutation) == self.ncomponents:
                self.weights = self.weights[permutation]
                for i in range(self.ndims):
                    self.factor_matrices[i] = self.factor_matrices[i][:, permutation]
                return
            assert False, (
                "Number of elements in permutation does not match number of "
                "components in ktensor."
            )

        # TODO there is a relationship here between normalize and arrange that repeats
        #  tasks. Can this be made to be more efficient? ensure that factor matrices
        #  are normalized
        self.normalize()

        # sort
        p = np.argsort(self.weights)[::-1]
        self.weights = self.weights[p]
        for i in range(self.ndims):
            self.factor_matrices[i] = self.factor_matrices[i][:, p]

        # absorb the weights into one factor, optional
        if weight_factor is not None:
            self.factor_matrices[weight_factor] *= self.weights
            self.weights = np.ones_like(self.weights)

        return

    def copy(self) -> ktensor:
        """
        Make a deep copy of a :class:`pyttb.ktensor`.

        Returns
        -------
        Copy of original ktensor.

        Examples
        --------
        Create a random :class:`pyttb.ktensor` with weights of 1:

        >>> np.random.seed(1)
        >>> K = ttb.ktensor.from_function(np.random.random_sample, (2, 3, 4), 2)
        >>> print(K)  # doctest: +ELLIPSIS
        ktensor of shape (2, 3, 4) with order F
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

        Create a copy of the :class:`pyttb.ktensor` and change the weights:

        >>> K2 = K.copy()
        >>> K2.weights = np.array([2.0, 3.0])
        >>> print(K2)  # doctest: +ELLIPSIS
        ktensor of shape (2, 3, 4) with order F
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

        Show that the original :class:`pyttb.ktensor` is unchanged:

        >>> print(K)  # doctest: +ELLIPSIS
        ktensor of shape (2, 3, 4) with order F
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
        """
        return ttb.ktensor(self.factor_matrices, self.weights, copy=True)

    def __deepcopy__(self, memo):
        """Return deep copy of ktensor."""
        return self.copy()

    def double(self, immutable: bool = False) -> np.ndarray:
        """
        Convert :class:`pyttb.ktensor` to :class:`numpy.ndarray`.

        Parameters
        ----------
        immutable: Whether or not the returned data cam be mutated. May enable
            additional optimizations.

        Returns
        -------
        Array of re-assembled ktensor.

        Examples
        --------
        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor(factor_matrices, weights)
        >>> K.double()
        array([[29., 39.],
               [63., 85.]])
        >>> type(K.double())
        <class 'numpy.ndarray'>
        """
        return self.full().double(immutable)

    def extract(self, idx: int | tuple | list | np.ndarray | None = None) -> ktensor:
        """Create a new :class:`pyttb.ktensor` with only the specified components.

        Parameters
        ----------
        idx:
            Index set of components to extract. It should be the case that
            `idx` is a subset of [0,...,`self.ncomponents`]. If this
            parameter is None or is empty, a copy of the
            :class:`pyttb.ktensor` is returned.

        Returns
        -------
        Subset of original ktensor.

        Examples
        --------
        Create a :class:`pyttb.ktensor`:

        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> K = ttb.ktensor([fm0, fm1], weights)
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Create a new :class:`pyttb.ktensor`, extracting only the second
        component from each factor of the original :class:`pyttb.ktensor`:

        >>> K.extract([1])
        ktensor of shape (2, 2) with order F
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
            return self.copy()

        if isinstance(idx, (int, tuple, list, np.ndarray)):
            if isinstance(idx, int):
                components = np.array([idx])
            else:
                components = np.asarray(idx)
            if len(components) == 0 or len(components) > self.ncomponents:
                assert False, (
                    f"Number of components requested is not valid: {len(components)} "
                    f"(should be in [1,...,{self.ncomponents}])."
                )
            else:
                # check that all requested component indices are valid
                invalid_entries = []
                for component in components:
                    if component not in range(self.ncomponents):
                        invalid_entries.append(component)
                if len(invalid_entries) > 0:
                    assert False, (
                        f"Invalid component indices to be extracted: "
                        f"{np_to_python(invalid_entries)} "
                        f"not in range({self.ncomponents})"
                    )
                new_weights = self.weights[components]
                new_factor_matrices = []
                for i in range(self.ndims):
                    new_factor_matrices.append(self.factor_matrices[i][:, components])
                return ttb.ktensor(new_factor_matrices, new_weights)
        else:
            assert False, "Input parameter must be an int, tuple, list or numpy.ndarray"

    def fixsigns(self, other: ktensor | None = None) -> ktensor:  # noqa: PLR0912
        """Change the elements of a :class:`pyttb.ktensor` in place.

        Update so that the
        largest magnitude entries for each column vector in each factor
        matrix are positive, provided that the sign on pairs of vectors in a
        rank-1 component can be flipped.

        Parameters
        ----------
        other:
            If not None, returns a version of the :class:`pyttb.ktensor`
            where some of the signs of the columns of the factor matrices have
            been flipped to better align with `other`. In not None, both
            :class:`pyttb.ktensor` objects are first normalized (using
            :func:`~pyttb.ktensor.normalize`).

        Returns
        -------
        Self for chained operations.

        Examples
        --------
        Create a :class:`pyttb.ktensor` with negative large magnitude entries:

        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> K = ttb.ktensor([fm0, fm1], weights)
        >>> K.factor_matrices[0][1, 1] = -K.factor_matrices[0][1, 1]
        >>> K.factor_matrices[1][1, 1] = -K.factor_matrices[1][1, 1]
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[ 1.  2.]
         [ 3. -4.]]
        factor_matrices[1] =
        [[ 5.  6.]
         [ 7. -8.]]

        Fix the signs of the largest magnitude entries:

        >>> print(K.fixsigns())
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[ 1. -2.]
         [ 3.  4.]]
        factor_matrices[1] =
        [[ 5. -6.]
         [ 7.  8.]]

        Fix the signs using another :class:`pyttb.ktensor`:

        >>> K = ttb.ktensor([fm0, fm1], weights)
        >>> K2 = K.copy()
        >>> K2.factor_matrices[0][1, 1] = -K2.factor_matrices[0][1, 1]
        >>> K2.factor_matrices[1][1, 1] = -K2.factor_matrices[1][1, 1]
        >>> K = K.fixsigns(K2)
        >>> print(K)  # doctest: +ELLIPSIS
        ktensor of shape (2, 2) with order F
        weights=[27.2029... 89.4427...]
        factor_matrices[0] =
        [[ 0.3162... -0.4472...]
         [ 0.9486... -0.8944...]]
        factor_matrices[1] =
        [[ 0.5812... -0.6...]
         [ 0.8137... -0.8...]]
        """
        if other is None:
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

        if not isinstance(other, ktensor):
            assert False, "other must be a ktensor"
        # Makes typing happy https://github.com/python/mypy/issues/4805
        other_tensor = other

        self.normalize()
        other_tensor = other_tensor.normalize()

        N = self.ndims
        RA = self.ncomponents
        RB = other_tensor.ncomponents

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
                    self.factor_matrices[n][:, r].T
                    @ other_tensor.factor_matrices[n][:, r]
                )

            # Sort the sign scores.
            sort_idx = np.argsort(sgn_score)
            sort_sgn_score = sgn_score.copy("K")[sort_idx]

            # Determine the number of scores that should be flipped.
            breakpt = np.nonzero(sort_sgn_score < 0)[-1]

            # If nothing needs to be flipped, then move on the the next component.
            if len(breakpt) == 0:
                continue
            breakpt = breakpt[-1]

            # Need to flip signs in pairs. If we don't have an even number of
            # negative sign scores, then we need to decide to do one fewer or one
            # more.
            if np.mod(breakpt + 1, 2) == 0:
                endpt = breakpt + 1
            else:
                warnings.warn(f"Trouble fixing signs for mode {r}")
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

    def to_tensor(self) -> ttb.tensor:
        """Convert to tensor.

        Same as :meth:`pyttb.ktensor.full`.
        """
        return self.full()

    def full(self) -> ttb.tensor:
        """
        Convert a :class:`pyttb.ktensor` to a :class:`pyttb.tensor`.

        Returns
        -------
        Re-assembled dense tensor.

        Examples
        --------
        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> K = ttb.ktensor([fm0, fm1], weights)
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]
        >>> print(K.full())  # doctest: +NORMALIZE_WHITESPACE
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[29. 39.]
         [63. 85.]]
        <BLANKLINE>
        """

        def min_split_dims(dims: tuple[int, ...]):
            """Return Minimum split dimensions.

            Solve
              min_{i in range(1,d)}  product(dims[:i]) + product(dims[i:])
            to minimize the memory footprint of the intermediate matrix.
            """
            sum_of_prods = [
                prod(dims[:i]) + prod(dims[i:]) for i in range(1, len(dims))
            ]
            i_min = np.argmin(sum_of_prods) + 1  # note range above starts at 1
            return i_min

        i_split = min_split_dims(self.shape)
        data = (
            ttb.khatrirao(*self.factor_matrices[:i_split], reverse=True) * self.weights
        ) @ ttb.khatrirao(*self.factor_matrices[i_split:], reverse=True).T
        # Copy needed to ensure F order. Transpose above means both elements are
        # different layout. If originally in C order can save on this copy.
        return ttb.tensor(data, self.shape, copy=True)

    def to_tenmat(
        self,
        rdims: np.ndarray | None = None,
        cdims: np.ndarray | None = None,
        cdims_cyclic: Literal["fc"] | Literal["bc"] | Literal["t"] | None = None,
        copy: bool = True,
    ) -> ttb.tenmat:
        """Construct a :class:`pyttb.tenmat` from a :class:`pyttb.ktensor`.

        Parameters
        ----------
        rdims:
            Mapping of row indices.
        cdims:
            Mapping of column indices.
        cdims_cyclic:
            When only rdims is specified maps a single rdim to the rows and
                the remaining dimensions span the columns. _fc_ (forward cyclic)
                in the order range(rdims,self.ndims()) followed by range(0, rdims).
                _bc_ (backward cyclic) range(rdims-1, -1, -1) then
                range(self.ndims(), rdims, -1).
        copy:
            Whether to make a copy of provided data or just reference it.

        Notes
        -----
        Forward cyclic is defined by Kiers [1]_ and backward cyclic is defined by
            De Lathauwer, De Moor, and Vandewalle [2]_.

        References
        ----------
        .. [1] KIERS, H. A. L. 2000. Towards a standardized notation and terminology
               in multiway analysis. J. Chemometrics 14, 105-122.
        .. [2] DE LATHAUWER, L., DE MOOR, B., AND VANDEWALLE, J. 2000b. On the best
               rank-1 and rank-(R1, R2, ... , RN ) approximation of higher-order
               tensors. SIAM J. Matrix Anal. Appl. 21, 4, 1324-1342.

        Examples
        --------
        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> K = ttb.ktensor([fm0, fm1], weights)
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]
        >>> K.full()  # doctest: +NORMALIZE_WHITESPACE
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[29. 39.]
         [63. 85.]]
        >>> K.to_tenmat(np.array([0]))  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[29. 39.]
         [63. 85.]]
        """
        # Simplest but slightly less efficient solution
        return self.full().to_tenmat(rdims, cdims, cdims_cyclic, copy)

    def innerprod(
        self, other: ttb.tensor | ttb.sptensor | ktensor | ttb.ttensor
    ) -> float:
        """
        Efficient inner product with a :class:`pyttb.ktensor`.

        Efficiently computes the inner product between two tensors, `self`
            and `other`.  If other is a :class:`pyttb.ktensor`, the inner
            product is computed using inner products of the factor matrices.
            Otherwise, the inner product is computed using the `ttv` (tensor
            times vector) of `other` with all of the columns of
            `self.factor_matrices`.

        Parameters
        ----------
        other:
            Tensor with which to compute the inner product.

        Returns
        -------
        Innerproduct value.

        Examples
        --------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> print(K.innerprod(K))
        96.0
        """
        if self.shape != other.shape:
            assert False, "Innerprod can only be computed for tensors of the same size"

        if isinstance(other, ktensor):
            M = np.outer(self.weights, other.weights)
            for i in range(self.ndims):
                M = M * (self.factor_matrices[i].T @ other.factor_matrices[i])
            return float(np.sum(np.sum(M)))

        if isinstance(other, (ttb.sptensor, ttb.tensor, ttb.ttensor)):
            res = 0.0
            for r in range(self.ncomponents):
                vecs = []
                for n in range(self.ndims):
                    vecs.append(self.factor_matrices[n][:, r])
                res = res + self.weights[r] * other.ttv(vecs)
            return float(res)
        raise ValueError(
            f"Unsupported type for inner product with ktensor. Received {type(other)}"
        )

    def isequal(self, other: ttb.ktensor) -> bool:
        """
        Equal comparator for :class:`pyttb.ktensor` objects.

        Parameters
        ----------
        other:
            :class:`pyttb.ktensor` with which to compare.

        Examples
        --------
        >>> K1 = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> weights = np.ones((2,))
        >>> factor_matrices = [np.ones((2, 2)), np.ones((3, 2)), np.ones((4, 2))]
        >>> K2 = ttb.ktensor(factor_matrices, weights)
        >>> print(K1.isequal(K2))
        True
        """
        if not isinstance(other, ktensor):
            return False
        if self.ncomponents != other.ncomponents:
            return False
        if (self.weights != other.weights).any():
            return False
        for k in range(self.ndims):
            if not np.array_equal(self.factor_matrices[k], other.factor_matrices[k]):
                return False
        return True

    @overload
    def issymmetric(
        self, return_diffs: Literal[False]
    ) -> bool: ...  # pragma: no cover see coveragepy/issues/970

    @overload
    def issymmetric(
        self, return_diffs: Literal[True]
    ) -> tuple[bool, np.ndarray]: ...  # pragma: no cover see coveragepy/issues/970

    def issymmetric(self, return_diffs: bool = False) -> bool | tuple[bool, np.ndarray]:
        """Return True if :class:`pyttb.ktensor` is symmetric for every permutation.

        Parameters
        ----------
        return_diffs:
            If True, returns the matrix of the norm of the differences
            between the factor matrices.

        Returns
        -------
        Answer and optionally matrix of the norm of the differences\
            between the factor matrices

        Examples
        --------
        Create a :class:`pyttb.ktensor` that is symmetric and test if it is
        symmetric:

        >>> K = ttb.ktensor.from_function(np.ones, (3, 3, 3), 2)
        >>> print(K.issymmetric())
        True

        Create a :class:`pyttb.ktensor` that is not symmetric and return the
        differences:

        >>> weights = np.array([1., 2.])
        >>> fm0 = np.array([[1., 2.], [3., 4.]])
        >>> fm1 = np.array([[5., 6.], [7., 8.]])
        >>> K2 = ttb.ktensor([fm0, fm1], weights)
        >>> issym, diffs = K2.issymmetric(return_diffs=True)
        >>> print(diffs)
        [[0. 8.]
         [0. 0.]]
        """
        diffs = np.zeros((self.ndims, self.ndims))
        for i in range(self.ndims):
            for j in range(i + 1, self.ndims):
                if self.factor_matrices[i].shape != self.factor_matrices[j].shape:
                    diffs[i, j] = np.inf
                elif np.array_equal(self.factor_matrices[i], self.factor_matrices[j]):
                    diffs[i, j] = 0
                else:
                    diffs[i, j] = np.linalg.norm(
                        self.factor_matrices[i] - self.factor_matrices[j]
                    )
        issym = (diffs == 0).all()

        if return_diffs:
            return issym, diffs
        return issym

    def mask(self, W: ttb.tensor | ttb.sptensor) -> np.ndarray:
        """Extract :class:`pyttb.ktensor` values as specified by `W`.

        `W` is a
        :class:`pyttb.tensor` or :class:`pyttb.sptensor` containing
        only values of zeros (0) and ones (1). The values in the
        :class:`pyttb.ktensor` corresponding to the indices for the
        ones (1) in `W` will be returned as a column vector.

        Parameters
        ----------
        W:
            Mask tensor to apply to ktensor.

        Returns
        -------
        Extracted values in a column vector (array).

        Examples
        --------
        Create a :class:`pyttb.ktensor`:

        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> K = ttb.ktensor([fm0, fm1], weights)

        Create a mask :class:`pyttb.tensor` and extract the elements of the
        :class:`pyttb.ktensor` using the mask:

        >>> W = ttb.tensor(np.array([[0, 1], [1, 0]]))
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

    def mttkrp(
        self, U: ktensor | Sequence[np.ndarray], n: int | np.integer
    ) -> np.ndarray:
        """
        Matricized tensor times Khatri-Rao product for :class:`pyttb.ktensor`.

        Efficiently calculates the matrix product of the n-mode matricization
        of the `ktensor` with the Khatri-Rao product of all entries in U,
        a :class:`list` of factor matrices, except the nth.

        Parameters
        ----------
        U:
            Factor matrices.
        n:
            Multiply by all modes except n.

        Returns
        -------
        Computed result.

        Examples
        --------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> U = [np.ones((2, 2)), np.ones((3, 2)), np.ones(((4, 2)))]
        >>> print(K.mttkrp(U, 0))
        [[24. 24.]
         [24. 24.]]
        """
        U = get_mttkrp_factors(U, n, self.ndims)

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
        return to_memory_order(self.factor_matrices[n] @ W, self.order)

    @property
    def ncomponents(self) -> int:
        """Number of columns in each factor matrix for the :class:`pyttb.ktensor`.

        Examples
        --------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> print(K.ncomponents)
        2
        """
        return len(self.weights)

    @property
    def ndims(self) -> int:
        """Number of dimensions of the :class:`pyttb.ktensor`.

        Examples
        --------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> print(K.ndims)
        3
        """
        return len(self.factor_matrices)

    def norm(self) -> float:
        """Compute the norm of a :class:`pyttb.ktensor`.

        Frobenius norm, or square root of the sum of
        squares of entries.

        Examples
        --------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> K.norm()  # doctest: +ELLIPSIS
        9.79795897...
        """
        # Compute the matrix of correlation coefficients
        coefMatrix = self.weights[:, None] @ self.weights[None, :]
        for f in self.factor_matrices:
            coefMatrix = coefMatrix * (f.T @ f)
        return float(np.sqrt(np.abs(np.sum(coefMatrix))))

    def normalize(
        self,
        weight_factor: int | Literal["all"] | None = None,
        sort: bool | None = False,
        normtype: float = 2,
        mode: int | None = None,
    ) -> ktensor:
        """Normalize the columns of the factor matrices in place.

        Optionally absorb the weights into desired normalized factors.

        Parameters
        ----------
        weight_factor:
            Absorb the weights into one or more factors. If "all", absorb
            weight equally across all factors. If `int`, absorb weight into a
            single dimension (value must be in range(self.ndims)).
        sort:
            Sort the columns in descending order of the weights.
        normtype:
            Order of the norm (see :func:`numpy.linalg.norm` for possible
            values).
        mode:
            Index of factor matrix to normalize. A value of `None` means
            normalize all factor matrices.

        Returns
        -------
        Self for chained operations.

        Examples
        --------
        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> print(K.normalize())  # doctest: +ELLIPSIS
        ktensor of shape (2, 3, 4) with order F
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
                return self
            assert False, (
                "Parameter single_factor is invalid; index must be an int in "
                "range of number of dimensions"
            )

        # ensure that all factor_matrices are normalized
        for mode_idx in range(self.ndims):
            for r in range(self.ncomponents):
                tmp = np.linalg.norm(self.factor_matrices[mode_idx][:, r], ord=normtype)
                if tmp > 0:
                    self.factor_matrices[mode_idx][:, r] = (
                        1.0 / tmp * self.factor_matrices[mode_idx][:, r]
                    )
                self.weights[r] = self.weights[r] * tmp

        # check that all weights are positive,
        # flip sign of columns in first factor matrix if negative weight found
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
        elif weight_factor is not None and weight_factor in range(self.ndims):
            # single factor
            self.factor_matrices[weight_factor] = self.factor_matrices[
                weight_factor
            ] @ np.diag(self.weights)
            self.weights = np.ones(self.weights.shape)

        if sort:
            if self.ncomponents > 1:
                # indices of string in descending order
                p = np.argsort(self.weights)[::-1]
                self.arrange(permutation=p)

        return self

    def nvecs(self, n: int, r: int, flipsign: bool = True) -> np.ndarray:
        """
        Compute the leading mode-n vectors of the ktensor.

        Computes the `r` leading eigenvectors of Xn*Xn.T (where Xn is the
        mode-`n` matricization/unfolding of self), which provides information
        about the mode-n fibers. In two-dimensions, the `r` leading mode-1
        vectors are the same as the `r` left singular vectors and the `r`
        leading mode-2 vectors are the same as the `r` right singular
        vectors. By default, this method computes the top `r` eigenvectors
        of Xn*Xn.T.

        Parameters
        ----------
        n:
            Mode for tensor matricization.
        r:
            Number of eigenvectors to compute and use.
        flipsign:
            If True, make each column's largest element positive.

        Returns
        -------
        Computed eigenvectors.

        Examples
        --------
        Compute single eigenvector for dimension 0:

        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
        >>> nvecs1 = K.nvecs(0, 1)
        >>> print(nvecs1)  # doctest: +ELLIPSIS
        [[0.70710678...]
         [0.70710678...]]

        Compute first 2 leading eigenvectors for dimension 0:

        >>> nvecs2 = K.nvecs(0, 2)
        >>> print(nvecs2)  # doctest: +ELLIPSIS
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
            logging.debug(
                "Greater than or equal to ktensor.shape[n] - 1 eigenvectors requires "
                "cast to dense to solve"
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

    def permute(self, order: OneDArray) -> ktensor:
        """
        Permute :class:`pyttb.ktensor` dimensions.

        Rearranges the dimensions of a :class:`pyttb.ktensor` so that they are
        in the order specified by `order`. The corresponding ktensor has the
        same components as `self` but the order of the subscripts needed to
        access any particular element is rearranged as specified by `order`.

        Parameters
        ----------
        order:
            Permutation of [0,...,self.ndimensions].

        Returns
        -------
        Permuted :class:`pyttb.ktensor`.

        Examples
        --------
        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor(factor_matrices, weights)
        >>> print(K)
        ktensor of shape (2, 2) with order F
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
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[5. 6.]
         [7. 8.]]
        factor_matrices[1] =
        [[1. 2.]
         [3. 4.]]
        """
        order = parse_one_d(order)
        # Check that the permutation is valid
        if tuple(range(self.ndims)) != tuple(sorted(order.tolist())):
            assert False, "Invalid permutation"

        return ttb.ktensor([self.factor_matrices[i] for i in order], self.weights)

    def redistribute(self, mode: int) -> ktensor:
        """Distribute weights of a :class:`pyttb.ktensor` to the specified mode.

        The redistribution is performed in place.

        Parameters
        ----------
        mode:
            Must be value in [0,...self.ndims].

        Returns
        -------
        Self for chained operations.

        Example
        -------
        Create a :class:`pyttb.ktensor`:

        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor(factor_matrices, weights)
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Distribute weights of that :class:`pyttb.ktensor` to mode 0:

        >>> K.redistribute(0)
        ktensor of shape (2, 2) with order F
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
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of a :class:`pyttb.ktensor`.

        Returns the lengths of all dimensions of the :class:`pyttb.ktensor`.

        """
        return tuple(f.shape[0] for f in self.factor_matrices)

    def score(
        self,
        other: ktensor,
        weight_penalty: bool = True,
        threshold: float | None = None,
        greedy: bool = True,
    ) -> tuple[float, ktensor, bool, np.ndarray]:
        """Check if two :class:`pyttb.ktensor` with the same shape match.

        Matching is defined as follows. If `self` and `other` are single-
        component :class:`pyttb.ktensor` instances that have been normalized
        so that their weights are `self.weights` and `other.weights`, and their
        factor matrices are single column vectors containing [a1,a2,...,an] and
        [b1,b2,...bn], respectively, then the score is defined as

            score = penalty * (a1.T*b1) * (a2.T*b2) * ... * (an.T*bn),

        where the penalty is defined by the weights such that

            max_weights = max(self.weights, other.weights)

            penalty = 1 - abs(self.weights - other.weights) / max_weights.

        The score of multi-component :class:`pyttb.ktensor` instances is a
        normalized sum of the scores across the best permutation of the
        components of `self`. `self` can have more components than `other`;
        any extra components are ignored in terms of the matching score.

        Parameters
        ----------
        other:
            :class:`pyttb.ktensor` with which to match.
        weight_penalty:
            Flag indicating whether or not to consider the weights in the
            calculations.
        threshold:
            Threshold specified in the formula above for determining a match.
            (defaults to: 0.99**self.ndims)
        greedy:
            Flag indicating whether or not to consider all possible matchings
            (exponentially expensive) or just do a greedy matching.

        Returns
        -------
        Score:
            Between 0 and 1.
        Copy of `self`:
            Which has been normalized and permuted to best match
            `other`.
        Flag:
            Indicating a match according to a user-specified threshold.
        Permutation:
            (i.e. array of indices of the modes of self) of the
            components of `self` that was used to best match `other`.

        Examples
        --------
        Create two :class:`pyttb.ktensor` instances and compute the score
        between them:

        >>> factors = [
        ...     np.ones((3, 3)) + 0.1,
        ...     np.ones((4, 3)) + 0.2,
        ...     np.ones((5, 3)) + 0.3,
        ... ]
        >>> weights = np.array([2.0, 1.0, 3.0])
        >>> K = ttb.ktensor(factors, weights)
        >>> factors_2 = [
        ...     np.ones((3, 2)) + 0.1,
        ...     np.ones((4, 2)) + 0.2,
        ...     np.ones((5, 2)) + 0.3,
        ... ]
        >>> weights_2 = np.array([2.0, 4.0])
        >>> K2 = ttb.ktensor(factors_2, weights_2)
        >>> score, Kperm, flag, perm = K.score(K2)
        >>> print(np.isclose(score, 0.875))
        True
        >>> print(perm)
        [0 2 1]

        Compute score without using weights:

        >>> score, Kperm, flag, perm = K.score(K2, weight_penalty=False)
        >>> print(np.isclose(score, 1.0))
        True
        >>> print(perm)
        [0 1 2]
        """
        assert greedy, (
            "Not yet implemented. Only greedy method is implemented currently."
        )

        assert isinstance(other, ktensor), "The first input should be a ktensor"

        assert self.shape == other.shape, "Size mismatch"

        if threshold is None:
            threshold = 0.99**self.ndims

        assert 0.0 <= threshold <= 1.0, "Threshold must be in range [0.0, 1.0]"

        # Set-up
        N = self.ndims
        RA = self.ncomponents
        RB = other.ncomponents

        # We're matching components in A to B
        if RA < RB:
            assert False, "Tensor A must have at least as many components as tensor B"

        # Make sure columns of factor matrices are normalized
        A = self.copy().normalize()
        B = other.copy().normalize()

        # Compute all possible vector-vector congruences.

        # Compute every pair for each mode
        Cbig = ttb.tensor(np.zeros((RA, RB, N), order=self.order))
        for n in range(N):
            Cbig[:, :, n] = np.abs(A.factor_matrices[n].T @ B.factor_matrices[n])

        # Collapse across all modes using the product
        collapsed = cast("ttb.ttensor", Cbig.collapse(np.array([2]), np.prod))
        C = collapsed.double()

        # Calculate penalty based on differences in the weights
        # Note that we are assuming the the weights are positive because the
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
            best_score = 0.0
            for _ in range(RB):
                flatten_C = C.reshape(prod(C.shape), order=self.order)
                idx = np.argmax(flatten_C)
                ij = tt_ind2sub((RA, RB), np.array(idx, dtype=int), order=self.order)
                best_score = best_score + C[ij[0], ij[1]]
                C[ij[0], :] = -10
                C[:, ij[1]] = -10
                best_perm[ij[1]] = ij[0]
            best_score = best_score / RB
            flag = best_score <= threshold

            # Rearrange the components of A according to the best matching
            foo = np.arange(RA)
            tf = np.isin(foo, best_perm)
            best_perm[RB : RA + 1] = foo[~tf]
            A.arrange(permutation=best_perm)
            return best_score, A, flag, best_perm
        raise ValueError("Unsupported score option")  # pragma: no cover

    def symmetrize(self) -> ktensor:
        """
        Symmetrize a :class:`pyttb.ktensor` in all modes.

        Symmetrize a :class:`pyttb.ktensor` with respect to all modes so that
        the resulting :class:`pyttb.ktensor` is symmetric with respect to any
        permutation of indices.

        Returns
        -------
        :class:`pyttb.ktensor`

        Examples
        --------
        Create a :class:`pyttb.ktensor`:

        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor(factor_matrices, weights)
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Make the factor matrices of the :class:`pyttb.ktensor` symmetric with
        respect to any permutation of the factor matrices:

        >>> K1 = K.symmetrize()
        >>> print(K1)  # doctest: +ELLIPSIS
        ktensor of shape (2, 2) with order F
        weights=[1. 1.]
        factor_matrices[0] =
        [[2.3404... 4.9519...]
         [4.5960... 8.0124...]]
        factor_matrices[1] =
        [[2.3404... 4.9519...]
         [4.5960... 8.0124...]]
        """
        # Check tensor dimensions for compatibility with symmetrization
        assert np.array_equal(self.shape, self.shape[0] * np.ones(self.ndims)), (
            "Tensor is not cubic -- cannot be symmetrized"
        )

        # Distribute lambda evenly into factors
        K = self.copy()
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

        return ttb.ktensor([V.copy("K") for i in range(K.ndims)], weights)

    def tolist(self, mode: int | None = None) -> list[np.ndarray]:
        """Convert :class:`pyttb.ktensor` to a list of factor matrices.

        Eevenly
        distributes the weights across factors. Optionally absorb the
        weights into a single mode.

        Parameters
        ----------
        mode:
            Index of factor matrix to absorb all of the weights.

        Returns
        -------
        Distributed factor matrices.

        Examples
        --------
        Create a :class:`pyttb.ktensor` of all ones:

        >>> weights = np.array([1.0, 2.0])
        >>> fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> factor_matrices = [fm0, fm1]
        >>> K = ttb.ktensor(factor_matrices, weights)
        >>> print(K)
        ktensor of shape (2, 2) with order F
        weights=[1. 2.]
        factor_matrices[0] =
        [[1. 2.]
         [3. 4.]]
        factor_matrices[1] =
        [[5. 6.]
         [7. 8.]]

        Spread weights equally to all factors and return list of factor
        matrices:

        >>> fm_list = K.tolist()
        >>> for fm in fm_list:
        ...     print(fm)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[1. 2.8284...]
         [3. 5.6568...]]
        [[ 5. 8.4852...]
         [ 7. 11.313...]]

        Shift weight to single factor matrix and return list of factor
        matrices:

        >>> fm_list = K.tolist(0)
        >>> for fm in fm_list:
        ...     print(fm)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[ 8.6023... 40. ]
         [25.8069... 80. ]]
        [[0.5812... 0.6...]
         [0.8137... 0.8...]]
        """
        if mode is not None:
            if isinstance(mode, int) and mode in range(self.ndims):
                self.normalize(mode)
                return self.factor_matrices.copy()
            assert False, "Input parameter'mode' must be in the range of self.ndims"

        # all weights are equal to 1
        if np.array_equal(self.weights, np.ones(self.weights.shape)):
            return self.factor_matrices.copy()

        lsgn = np.sign(self.weights)
        D = np.diag(np.power(np.fabs(self.weights), 1.0 / self.ndims))
        factor_matrices = self.factor_matrices.copy()
        factor_matrices[0] = factor_matrices[0] @ np.diag(lsgn)
        for n in range(self.ndims):
            factor_matrices[n] = factor_matrices[n] @ D
        return factor_matrices

    def tovec(self, include_weights: bool = True) -> np.ndarray:
        """Convert :class:`pyttb.ktensor` to column vector.

        Optionally include
        or exclude the weights. The output of this method can be consumed by
        :meth:`from_vector`.

        Parameters
        ----------
        include_weights:
            Flag to specify whether or not to include weights in output.

        Returns
        -------
        The length of the column vector is\
            (sum(self.shape)+1)*self.ncomponents. The vector contains the\
            weights (if requested) stacked on top of each of the columns of\
            the factor_matrices in order.

        Examples
        --------
        Create a :class:`pyttb.ktensor` from a vector:

        >>> rank = 2
        >>> shape = (2, 3, 4)
        >>> data = np.arange(1, rank*sum(shape)+1)
        >>> weights = 2 * np.ones(rank)
        >>> weights_and_data = np.concatenate((weights, data), axis=0)
        >>> K = ttb.ktensor.from_vector(weights_and_data[:], shape, True)
        >>> print(K)
        ktensor of shape (2, 3, 4) with order F
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

        Create a :class:`pyttb.ktensor` from a vector of data extracted from
        another :class:`pyttb.ktensor`:

        >>> K2 = ttb.ktensor.from_vector(K.tovec(), shape, True)
        >>> print(K2)
        ktensor of shape (2, 3, 4) with order F
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

        return x

    def ttv(
        self,
        vector: Sequence[np.ndarray] | np.ndarray,
        dims: OneDArray | None = None,
        exclude_dims: OneDArray | None = None,
    ) -> float | ktensor:
        """
        Tensor times vector for a :class:`pyttb.ktensor`.

        Computes the product of a :class:`pyttb.ktensor` with a vector (i.e.,
        np.array).  If `dims` is an integer, it specifies the dimension in the
        :class:`pyttb.ktensor` along which the vector is multiplied.
        If the shape of the vector is = (I,), then the length of dimension
        `dims` of the :class:`pyttb.ktensor` must be  I.  Note that the number
        of dimensions of the returned :class:`pyttb.ktensor` is 1 less than
        the dimension of the :class:`pyttb.ktensor` used in the
        multiplication because dimension `dims` is removed.

        If `vector` is a :class:`list` of np.array instances, the
        :class:`pyttb.ktensor` is multiplied with each vector in the list. The
        products are computed sequentially along all dimensions (or modes) of
        the :class:`pyttb.ktensor`, and thus the list must contain `self.ndims`
        vectors.

        When `dims` is not None, compute the products along the dimensions
        specified by `dims`. In this case, the number of products can be less
        than `self.ndims` and the order of the sequence does not need to match
        the order of the dimensions in the :class:`pyttb.ktensor`. Note that
        the number of vectors must match the number of dimensions provided,
        and the length of each vector must match the size of each dimension
        of the :class:`pyttb.ktensor` specified in `dims`.

        Parameters
        ----------
        vector:
            Vector to multiply by.
        dims:
            Dimension(s) along which to multiply.
                Exclusively provide dims or exclude_dims.
        exclude_dims:
            Multiply by all but excluded dimension(s).
                Exclusively provide dims or exclude_dims.

        Returns
        -------
        float or :class:`pyttb.ktensor`
            The number of dimensions of the returned :class:`pyttb.ktensor` is
            n-k, where n = self.ndims and k = number of vectors provided as
            input. If k == n, a scalar is returned.

        Examples
        --------
        Compute the product of a :class:`pyttb.ktensor` and a single vector
        (results in a :class:`pyttb.ktensor`):

        >>> rank = 2
        >>> shape = (2, 3, 4)
        >>> data = np.arange(1, rank * sum(shape) + 1)
        >>> weights = 2 * np.ones(rank)
        >>> weights_and_data = np.concatenate((weights, data), axis=0)
        >>> K = ttb.ktensor.from_vector(weights_and_data[:], shape, True)
        >>> K0 = K.ttv(np.array([1, 1, 1]), dims=1)  # compute along a single dimension
        >>> print(K0)
        ktensor of shape (2, 4) with order F
        weights=[36. 54.]
        factor_matrices[0] =
        [[1. 3.]
         [2. 4.]]
        factor_matrices[1] =
        [[11. 15.]
         [12. 16.]
         [13. 17.]
         [14. 18.]]

        Compute the product of a :class:`pyttb.ktensor` and a vector for each
        dimension (results in a `float`):

        >>> vec2 = np.array([1, 1])
        >>> vec3 = np.array([1, 1, 1])
        >>> vec4 = np.array([1, 1, 1, 1])
        >>> K1 = K.ttv([vec2, vec3, vec4])
        >>> print(K1)
        30348.0

        Compute the product of a :class:`pyttb.ktensor` and multiple vectors
        out of order (results in a :class:`pyttb.ktensor`):

        >>> K2 = K.ttv([vec4, vec3], np.array([2, 1]))
        >>> print(K2)
        ktensor of shape (2,) with order F
        weights=[1800. 3564.]
        factor_matrices[0] =
        [[1. 3.]
         [2. 4.]]
        """
        # Check vector is a list of vectors
        # if not place single vector as element in list
        if (
            len(vector) > 0
            and isinstance(vector, np.ndarray)
            and isinstance(vector.squeeze()[0], (int, float, np.int_, np.float64))
        ):
            return self.ttv([vector], dims, exclude_dims)

        # Get sorted dims and index for multiplicands
        dims, vidx = tt_dimscheck(self.ndims, len(vector), dims, exclude_dims)

        # Check that each multiplicand is the right size.
        for i in range(dims.size):
            if vector[vidx[i]].squeeze().shape != (self.shape[dims[i]],):
                assert False, (
                    f"Multiplicand is wrong size. Vector[{i}] was "
                    f"{vector[vidx[i]].squeeze().shape}"
                    f", but expected {(self.shape[dims[i]],)}."
                )

        # Figure out which dimensions will be left when we're done
        remdims = np.setdiff1d(range(self.ndims), dims)

        # Collapse dimensions that are being multiplied out
        new_weights = self.weights.copy("K")
        for i, dim in enumerate(dims):
            new_weights = new_weights * (
                self.factor_matrices[dim].T @ vector[vidx[i]].squeeze()
            )

        # Create final result
        if len(remdims) == 0:
            return sum(new_weights)

        factor_matrices = []
        for i in remdims:
            factor_matrices.append(self.factor_matrices[i])
        return ttb.ktensor(factor_matrices, new_weights, copy=False)

    def update(self, modes: OneDArray, data: np.ndarray) -> ktensor:
        """Update a :class:`pyttb.ktensor` in the specific dimensions.

        Updates with the
        values in `data` (in vector or matrix form). The value of `modes` must
        be a value in [-1,...,self.ndims]. If the Further, the number of elements in
        `data` must equal self.shape[modes] * self.ncomponents. The update is
        performed in place.

        Parameters
        ----------
        modes:
            List of dimensions to update; values must be in ascending order. If
            the first element of the list is -1, then update the weights. All
            other integer values values must be sorted and in
            [0,...,self.ndims].
        data:
            Data values to use in the update.

        Returns
        -------
        Self for chained operations.

        Examples
        --------
        Create a :class:`pyttb.ktensor` of all ones:

        >>> K = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)

        Create vectors for updating various factor matrices of the
        :class:`pyttb.ktensor`:

        >>> vec0 = 2 * np.ones(K.shape[0] * K.ncomponents)
        >>> vec1 = 3 * np.ones(K.shape[1] * K.ncomponents)
        >>> vec2 = 4 * np.ones(K.shape[2] * K.ncomponents)

        Update a single factor matrix:

        >>> K1 = K.copy()
        >>> K1 = K1.update(0, vec0)
        >>> print(K1)
        ktensor of shape (2, 3, 4) with order F
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
        ktensor of shape (2, 3, 4) with order F
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

        Update some but not all factor matrices:

        >>> K3 = K.copy()
        >>> vec_some = np.concatenate((vec0, vec2))
        >>> K3 = K3.update([0, 2], vec_some)
        >>> print(K3)
        ktensor of shape (2, 3, 4) with order F
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
        modes = parse_one_d(modes)
        assert np.all(modes[:-1] <= modes[1:]), (
            "Modes must be sorted in ascending order"
        )

        loc = 0  # Location in data array
        for k in modes:
            if k == -1:
                # update weights
                endloc = loc + self.ncomponents
                if len(data) < endloc:
                    assert False, "Data is too short"
                self.weights = data[loc:endloc].copy("K")
                loc = endloc
            elif k < self.ndims:
                # update factor matrix
                endloc = loc + self.shape[k] * self.ncomponents
                if len(data) < endloc:
                    assert False, "Data is too short"
                self.factor_matrices[k] = np.reshape(
                    data[loc:endloc].copy("K"),
                    (self.shape[k], self.ncomponents),
                    order=self.order,
                )
                loc = endloc
            else:
                assert False, f"Invalid mode: {k}"

        ## Check that we used all the data
        if loc != len(data):
            warnings.warn("Failed to consume all of the input data")

        return self

    def viz(  # noqa: PLR0912, PLR0913
        self,
        plots: tuple | list | None = None,
        show_figure: bool = True,
        normalize: bool = True,
        norm: int | float = 2,
        rel_weights: bool = True,
        rel_heights: tuple | list | None = None,
        rel_widths: tuple | list | None = None,
        horz_space: float | None = None,
        vert_space: float | None = None,
        left_space: float | None = None,
        right_space: float | None = None,
        top_space: float | None = None,
        bot_space: float | None = None,
        mode_titles: tuple | list | None = None,
        title=None,
    ) -> tuple[Figure, Axes]:
        """
        Visualize factors for :class:`pyttb.ktensor`.

        Parameters
        ----------
        plots:
            List of functions (one per mode) which visualize the respective vectors
            of a factor.
            Function for mode i must have signature `f(v_i,ax)` where
            `v_i` is :class:`numpy.ndarray` vector of dimension `n_i` and
            `ax` is a :class:`matplotlib.axes.Axes` on which to plot.
        show_figure:
            Boolean determining if the resulting figure should be shown.
        normalize:
            Boolean controlling whether to normalize factors and generate
            a compensating weight, then sort components by weight.
        norm:
            Norm used to normalize factors; 1 for 1-norm, 2 for 2-norm, etc.
        rel_weights:
            Boolean determining whether weights should be made relative by
            dividing by largest weight.
        rel_widths:
            List of numbers (one per mode) specifying relative widths of each
            plot column.
        rel_heights:
            List of numbers (one per component) specifying relative height of each
            plot row.
        horz/vert_space:
            Number determining amount of space between subplots
            (horizontally/vertically) as a fraction of the average axis width/height.
        left/right/top/bot_space:
            Extent of subplots as fraction of figure width or height.
        mode_titles:
            List of strings used as titles for each column (mode).
        title:
            String containing overall figure title.

        Returns
        -------
        fig:
            :class:`matplotlib.figure.Figure` handle for the generated figure
        axs:
            :class:`matplotlib.axes.Axes` for the generated figure

        Examples
        --------
        Set up a :class:`pyttb.ktensor` to plot:

        >>> np.random.seed(1)
        >>> K = ttb.ktensor.from_function(np.random.random_sample, (2, 3, 10), 2)

        Use plot K using default behavior K.viz()

        >>> fig, axs = K.viz(show_figure=False)  # doctest: +ELLIPSIS
        >>> plt.close(fig)

        Define a more realistic plot functions with x labels,
        control relative widths of each plot,
        and set mode titles.

        >>> def mode_1_plot(v, ax):
        ...     ax.bar([1, 2], v, width=0.2)
        ...     ax.set_xticks([1, 2], labels=["neutron", "electron"], rotation=45)
        >>> def mode_2_plot(v, ax):
        ...     ax.plot(np.arange(v.shape[0]), v)
        ...     ax.set_xlabel("$v$, [m/s]")
        >>> def mode_3_plot(v, ax):
        ...     ax.semilogx(np.logspace(-2, 2, v.shape[0]), v)
        ...     ax.set_xlabel("$E$, [kJ]")
        >>> plots = [mode_1_plot, mode_2_plot, mode_3_plot]
        >>> fig, axs = K.viz(
        ...     show_figure=False,
        ...     plots=plots,
        ...     rel_widths=[1, 2, 3],
        ...     horz_space=0.4,
        ...     left_space=0.2,
        ...     bot_space=0.2,
        ...     mode_titles=["Particle", "Velocity", "Energy"],
        ... )  # doctest: +ELLIPSIS
        >>> plt.close(fig)
        """

        def line_plot(v, ax):
            ax.plot(v)

        m = len(self.shape)  # number of modes
        r = self.ncomponents  # rank

        # booleans storing whether respective title/label should be rendered
        show_mode_titles = False
        show_title = False

        # check input validity
        if plots is None:
            plots = [line_plot] * m
        else:
            assert len(plots) == m, "Incorrect number of plot functions"
        if rel_widths is not None:
            assert len(rel_widths) == m, "Incorrect number of relative widths"
        if rel_heights is not None:
            assert len(rel_heights) == m, "Incorrect number of relative heights"
        if mode_titles is not None:
            assert len(mode_titles) == m, "Incorrect number of mode titles"
            show_mode_titles = True
        if title is not None:
            show_title = True
        if normalize:
            self.normalize(normtype=norm, sort=True)

        # compute factor weights (and optionally normalize)
        weights = self.weights.copy("K")
        weight_labels = [format(w, ".2e") for w in weights]
        if rel_weights:
            weights /= np.max(weights)
            weight_labels = [format(w, ".2f") for w in weights]

        # construct subplots
        fig, axs = plt.subplots(
            nrows=r,
            ncols=m,
            sharex="col",
            gridspec_kw={"width_ratios": rel_widths, "height_ratios": rel_heights},
        )

        # compute y lims for each mode
        y_lims = [[np.min(A), np.max(A)] for A in self.factor_matrices]

        # plot data on each axis
        for k in range(m):  # loop over modes
            is_first_col = k == 0
            U = self.factor_matrices[k].T  # r x n_k
            for j in range(r):  # loop over components (rows of U)
                is_first_row = j == 0
                is_last_row = j == r - 1
                # share y lims (unless user overrides with their plots)
                axs[j, k].set_ylim(*y_lims[k])
                # user defined plot (anything we do after may override them)
                plots[k](U[j], ax=axs[j, k])
                # render (or don't) titles/labels
                if is_first_col:
                    axs[j, k].set_ylabel(
                        weight_labels[j], rotation=0, ha="right", labelpad=10
                    )  # may be absolute or relative weight
                if is_first_row and show_mode_titles and mode_titles:
                    axs[j, k].set_title(mode_titles[k])
                # remove duplicates of xlabels
                if not is_last_row:
                    axs[j, k].set_xlabel(None)

        # figure title
        if show_title:
            fig.suptitle(title)

        # tune layout
        fig.subplots_adjust(
            wspace=horz_space,
            hspace=vert_space,
            left=left_space,
            right=right_space,
            top=top_space,
            bottom=bot_space,
        )

        if show_figure:
            fig.show()
        return fig, axs

    def __add__(self, other):
        """
        Binary addition for :class:`pyttb.ktensor`.

        Parameters
        ----------
        other: :class:`pyttb.ktensor`, required
            :class:`pyttb.ktensor` to add to `self`.

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        if isinstance(other, ttb.sumtensor):
            return other.__add__(self)
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
        return ttb.ktensor(factor_matrices, weights)

    def __neg__(self):
        """
        Unary minus (negative) for :class:`pyttb.ktensor` instances.

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        return ttb.ktensor(self.factor_matrices, -self.weights)

    def __pos__(self):
        """
        Unary plus (positive) for :class:`pyttb.ktensor` instances.

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        return self.copy()

    def __sub__(self, other):
        """
        Binary subtraction for :class:`pyttb.ktensor`.

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
        return ttb.ktensor(factor_matrices, weights)

    def __mul__(self, other):
        """Elementwise (including scalar) multiplication for :class:`pyttb.ktensor`.

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
            return ttb.ktensor(self.factor_matrices, other * self.weights)

        assert False, (
            "Multiplication by ktensors only allowed for scalars, tensors, or sptensors"
        )

    def __rmul__(self, other):
        """Elementwise (including scalar) multiplication for :class:`pyttb.ktensor`.

        Parameters
        ----------
        other: :class:`pyttb.tensor`, :class:`pyttb.sptensor`, float, int

        Returns
        -------
        :class:`pyttb.ktensor`
        """
        return self.__mul__(other)

    def __repr__(self):
        """Return string representation of a :class:`pyttb.ktensor`.

        Returns
        -------
        str:
        """
        s = f"ktensor of shape {self.shape} with order {self.order}\n"
        s += f"weights={self.weights}"
        if len(self.shape) == 0:
            s += "\nfactor_matrices=[]"
        else:
            for i, factor in enumerate(self.factor_matrices):
                s += f"\nfactor_matrices[{i}] =\n"
                s += str(factor)
        return s

    __str__ = __repr__


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
