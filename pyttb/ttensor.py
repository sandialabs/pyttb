"""Tucker Tensor Implementation"""

# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging
import textwrap
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy
from scipy import sparse

import pyttb as ttb
from pyttb import pyttb_utils as ttb_utils

ALT_CORE_ERROR = "TTensor doesn't support non-tensor cores yet. Only tensor/sptensor."


class ttensor:
    """
    TTENSOR Class for Tucker tensors (decomposed).
    """

    __slots__ = ("core", "factor_matrices")

    def __init__(
        self,
        core: Optional[Union[ttb.tensor, ttb.sptensor]] = None,
        factors: Optional[List[np.ndarray]] = None,
        copy: bool = True,
    ) -> None:
        """
        Construct an ttensor from fully defined core tensor and factor matrices.

        Parameters
        ----------
        core:
            Core of tucker tensor.
        factors:
            Factor matrices.
        copy:
            Whether to make a copy of provided data or just reference it.

        Returns
        -------
        Constructed tucker tensor.

        Examples
        --------
        Import required modules:

        >>> import pyttb as ttb
        >>> import numpy as np

        Set up input data
        # Create ttensor with explicit data description

        >>> core_values = np.ones((2,2,2))
        >>> core = ttb.tensor(core_values)
        >>> factors = [np.ones((1,2))] * len(core_values.shape)
        >>> K0 = ttb.ttensor(core, factors)
        """
        if core is None and factors is None:
            # Empty constructor
            # TODO explore replacing with typing protocol
            self.core: Union[ttb.tensor, ttb.sptensor] = ttb.tensor()
            self.factor_matrices: List[np.ndarray] = []
            return

        if core is None or factors is None:
            raise ValueError(
                "For non-empty ttensor both core and factors must be provided"
            )

        if isinstance(core, (ttb.tensor, ttb.sptensor)):
            if copy:
                self.core = core.copy()
                self.factor_matrices = deepcopy(factors)
            else:
                self.core = core
                self.factor_matrices = factors
        else:
            # TODO support any tensor type with supported ops
            raise ValueError(ALT_CORE_ERROR)
        self._validate_ttensor()
        return

    def copy(self) -> ttensor:
        """Make a deep copy of a :class:`pyttb.ttensor`.

        Returns
        -------
        Copy of original ttensor.

        Examples
        --------
        >>> core_values = np.ones((2,2,2))
        >>> core = ttb.tensor(core_values)
        >>> factors = [np.ones((1,2))] * len(core_values.shape)
        >>> first = ttb.ttensor(core, factors)
        >>> second = first
        >>> third = second.copy()
        >>> first.factor_matrices[0][0,0] = 2
        >>> first.factor_matrices[0][0,0] == second.factor_matrices[0][0,0]
        True
        >>> first.factor_matrices[0][0,0] == third.factor_matrices[0][0,0]
        False
        """
        return ttb.ttensor(self.core, self.factor_matrices, copy=True)

    def __deepcopy__(self, memo):
        return self.copy()

    def _validate_ttensor(self):
        """Verifies the validity of constructed ttensor"""
        # Confirm all factors are matrices
        for factor_idx, factor in enumerate(self.factor_matrices):
            if not isinstance(factor, (np.ndarray, sparse.coo_matrix)):
                raise ValueError(
                    f"Factor matrices must be numpy arrays but factor {factor_idx} "
                    f" was {type(factor)}"
                )
            if len(factor.shape) != 2:
                raise ValueError(
                    f"Factor matrix {factor_idx} has shape {factor.shape} and is not "
                    f"a matrix!"
                )

        # Verify size consistency
        core_order = len(self.core.shape)
        num_matrices = len(self.factor_matrices)
        if core_order != num_matrices:
            raise ValueError(
                f"CORE has order {core_order} but there are {num_matrices} factors"
            )
        for factor_idx, factor in enumerate(self.factor_matrices):
            if factor.shape[-1] != self.core.shape[factor_idx]:
                raise ValueError(
                    f"Factor matrix {factor_idx} does not have "
                    f"{self.core.shape[factor_idx]} columns"
                )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor this deconstruction represents."""
        return tuple(factor.shape[0] for factor in self.factor_matrices)

    def __repr__(self):  # pragma: no cover
        """
        String representation of a tucker tensor.

        Returns
        -------
        str
            Contains the core, and factor matrices as strings on different lines.
        """
        display_string = f"Tensor of shape: {self.shape}\n" f"\tCore is a\n"
        display_string += textwrap.indent(str(self.core), "\t")

        for factor_idx, factor in enumerate(self.factor_matrices):
            display_string += f"\tU[{factor_idx}] = \n"
            display_string += textwrap.indent(str(factor), "\t\t")
            display_string += "\n"
        return display_string

    __str__ = __repr__

    def to_tensor(self) -> ttb.tensor:
        """Convenience method to convert to tensor.
        Same as :meth:`pyttb.ttensor.full`
        """
        return self.full()

    def full(self) -> ttb.tensor:
        """Convert a ttensor to a (dense) tensor."""
        recomposed_tensor = self.core.ttm(self.factor_matrices)

        # There is a small chance tensor could be sparse so cast that to dense.
        if not isinstance(recomposed_tensor, ttb.tensor):
            recomposed_tensor = recomposed_tensor.to_tensor()
        return recomposed_tensor

    def double(self) -> np.ndarray:
        """
        Convert ttensor to an array of doubles

        Returns
        -------
        Copy of tensor data.
        """
        return self.full().double()

    @property
    def ndims(self) -> int:
        """
        Number of dimensions of a ttensor.

        Returns
        -------
        Number of dimensions of ttensor
        """
        return len(self.factor_matrices)

    def isequal(self, other: ttensor) -> bool:
        """
        Component equality for ttensors

        Parameters
        ----------
        other:
            TTensor to compare against.

        Returns
        -------
        True if ttensors decompositions are identical, false otherwise
        """
        if not isinstance(other, ttensor):
            return False
        if self.ndims != other.ndims:
            return False
        return self.core.isequal(other.core) and all(
            np.array_equal(this_factor, other_factor)
            for this_factor, other_factor in zip(
                self.factor_matrices, other.factor_matrices
            )
        )

    def __pos__(self):
        """
        Unary plus (+) for ttensors. Does nothing.

        Returns
        -------
        :class:`pyttb.ttensor`, copy of tensor
        """

        return self.copy()

    def __neg__(self):
        """
        Unary minus (-) for ttensors

        Returns
        -------
        :class:`pyttb.ttensor`, copy of tensor
        """
        return ttensor(-self.core, self.factor_matrices)

    def innerprod(
        self, other: Union[ttb.tensor, ttb.sptensor, ttb.ktensor, ttb.ttensor]
    ) -> float:
        """
        Efficient inner product with a ttensor

        Parameters
        ----------
        other:
            Tensor to take an innerproduct with.

        Returns
        -------
        Result of the innerproduct.
        """
        if isinstance(other, ttensor):
            if self.shape != other.shape:
                raise ValueError(
                    "ttensors must have same shape to perform an innerproduct, "
                    f" but this ttensor has shape {self.shape} and the other has "
                    f"{other.shape}"
                )
            if np.prod(self.core.shape) > np.prod(other.core.shape):
                # Reverse arguments so the ttensor with the smaller core comes first.
                return other.innerprod(self)
            W = []
            for this_factor, other_factor in zip(
                self.factor_matrices, other.factor_matrices
            ):
                W.append(this_factor.transpose().dot(other_factor))
            J = other.core.ttm(W)
            return self.core.innerprod(J)
        if isinstance(other, (ttb.tensor, ttb.sptensor)):
            if self.shape != other.shape:
                raise ValueError(
                    "ttensors must have same shape to perform an innerproduct, but "
                    f" this ttensor has shape {self.shape} and the other has "
                    f"{other.shape}"
                )
            if np.prod(self.shape) < np.prod(self.core.shape):
                Z: Union[ttb.tensor, ttb.sptensor] = self.full()
                return Z.innerprod(other)
            Z = other.ttm(self.factor_matrices, transpose=True)
            return Z.innerprod(self.core)
        if isinstance(other, ttb.ktensor):
            # Call ktensor implementation
            return other.innerprod(self)
        raise ValueError(
            f"Inner product between ttensor and {type(other)} is not supported"
        )

    def __mul__(self, other):
        """
        Element wise multiplication (*) for ttensors (only scalars supported)

        Parameters
        ----------
        other: float, int

        Returns
        -------
        :class:`pyttb.ttensor`
        """
        if isinstance(other, (float, int, np.number)):
            return ttensor(self.core * other, self.factor_matrices)
        raise ValueError(
            "This object cannot be multiplied by ttensor. Convert to full if trying to "
            "multiply ttensor by another tensor."
        )

    def __rmul__(self, other):
        """
        Element wise right multiplication (*) for ttensors (only scalars supported)

        Parameters
        ----------
        other: float, int

        Returns
        -------
        :class:`pyttb.ttensor`
        """
        if isinstance(other, (float, int, np.number)):
            return self.__mul__(other)
        raise ValueError("This object cannot be multiplied by ttensor")

    def ttv(
        self,
        vector: Union[List[np.ndarray], np.ndarray],
        dims: Optional[Union[int, np.ndarray]] = None,
        exclude_dims: Optional[Union[int, np.ndarray]] = None,
    ) -> Union[float, ttensor]:
        """
        TTensor times vector

        Parameters
        ----------
        vector:
            Vector to multiply by.
        dims:
            Dimensions to multiply in.
        exclude_dims:
            Alternative multiply by all dimensions but these.
        """
        if dims is None and exclude_dims is None:
            dims = np.array([])
        # TODO make helper function to check scalar since re-used many places
        elif isinstance(dims, (float, int)):
            dims = np.array([dims])

        if isinstance(exclude_dims, (float, int)):
            exclude_dims = np.array([exclude_dims])

        # Check that vector is a list of vectors,
        # if not place single vector as element in list
        if (
            len(vector) > 0
            and isinstance(vector, np.ndarray)
            and isinstance(vector[0], (int, float, np.int_, np.float64))
        ):
            return self.ttv([vector], dims, exclude_dims)

        # Get sorted dims and index for multiplicands
        dims, vidx = ttb_utils.tt_dimscheck(self.ndims, len(vector), dims, exclude_dims)

        # Check that each multiplicand is the right size.
        for i in range(dims.size):
            if vector[vidx[i]].shape != (self.shape[dims[i]],):
                raise ValueError("Multiplicand is wrong size")

        # Get remaining dimensions when we're done
        remdims = np.setdiff1d(np.arange(0, self.ndims), dims)

        # Create W to multiply with core, only populated remaining dims
        W = [np.empty(())] * self.ndims
        for i in range(dims.size):
            dim_idx = dims[i]
            W[dim_idx] = self.factor_matrices[dim_idx].transpose().dot(vector[vidx[i]])

        # Create new core
        newcore = self.core.ttv(W, dims)

        # Create final result
        if remdims.size == 0:
            assert not isinstance(newcore, (ttb.tensor, ttb.sptensor))
            return float(newcore)
        assert not isinstance(newcore, float)
        return ttensor(newcore, [self.factor_matrices[dim] for dim in remdims])

    def mttkrp(self, U: Union[ttb.ktensor, List[np.ndarray]], n: int) -> np.ndarray:
        """
        Matricized tensor times Khatri-Rao product for ttensors.

        Parameters
        ----------
        U:
            Array of matrices or ktensor
        n:
            Multiplies by all modes except n

        Returns
        -------
        :class:`numpy.ndarray`
        """
        # NOTE: MATLAB version calculates an unused R here

        W = [np.empty(())] * self.ndims
        if isinstance(U, ttb.ktensor):
            U = U.factor_matrices
        for i in range(0, self.ndims):
            if i == n:
                continue
            W[i] = self.factor_matrices[i].transpose().dot(U[i])

        Y = self.core.mttkrp(W, n)

        # Find each column of answer by multiplying by weights
        return self.factor_matrices[n].dot(Y)

    def norm(self) -> float:
        """
        Compute the norm of a ttensor.
        Returns
        -------
        Frobenius norm of Tensor.
        """
        if np.prod(self.shape) > np.prod(self.core.shape):
            V = []
            for factor in self.factor_matrices:
                V.append(factor.transpose().dot(factor))
            Y = self.core.ttm(V)
            tmp = Y.innerprod(self.core)
            return np.sqrt(tmp)
        return self.full().norm()

    def permute(self, order: np.ndarray) -> ttensor:
        """
        Permute :class:`pyttb.ttensor` dimensions.

        Rearranges the dimensions of a :class:`pyttb.ttensor` so that they are
        in the order specified by `order`. The corresponding ttensor has the
        same components as `self` but the order of the subscripts needed to
        access any particular element is rearranged as specified by `order`.

        Parameters
        ----------
        order:
            Permutation of [0,...,self.ndims].

        Returns
        -------
        Permuted :class:`pyttb.ttensor`.
        """
        if not np.array_equal(np.arange(0, self.ndims), np.sort(order)):
            raise ValueError("Invalid permutation")
        new_core = self.core.permute(order)
        new_u = [self.factor_matrices[idx] for idx in order]
        return ttensor(new_core, new_u)

    def ttm(
        self,
        matrix: Union[np.ndarray, List[np.ndarray]],
        dims: Optional[Union[float, np.ndarray]] = None,
        exclude_dims: Optional[Union[int, np.ndarray]] = None,
        transpose: bool = False,
    ) -> ttensor:
        """
        Tensor times matrix for ttensor

        Parameters
        ----------
        matrix:
            Matrix or matrices to multiple by
        dims:
            Dimensions to multiply against
        exclude_dims:
            Use all dimensions but these
        transpose:
            Transpose matrices during multiplication
        """
        if dims is None and exclude_dims is None:
            dims = np.arange(self.ndims)
        elif isinstance(dims, list):
            dims = np.array(dims)
        elif isinstance(dims, (float, int, np.generic)):
            dims = np.array([dims])

        if isinstance(exclude_dims, (float, int)):
            exclude_dims = np.array([exclude_dims])

        if not isinstance(matrix, list):
            return self.ttm([matrix], dims, exclude_dims, transpose)

        # Check that the dimensions are valid
        dims, vidx = ttb_utils.tt_dimscheck(self.ndims, len(matrix), dims, exclude_dims)

        # Determine correct size index
        size_idx = int(not transpose)

        # Check that each multiplicand is the right size.
        for i, dim in enumerate(dims):
            if matrix[vidx[i]].shape[size_idx] != self.shape[dim]:
                raise ValueError(f"Multiplicand {i} is wrong size")

        # Do the actual multiplications in the specified modes.
        new_u = self.factor_matrices.copy()
        for i, dim in enumerate(dims):
            if transpose:
                new_u[dim] = matrix[vidx[i]].transpose().dot(new_u[dim])
            else:
                new_u[dim] = matrix[vidx[i]].dot(new_u[dim])

        return ttensor(self.core, new_u)

    def reconstruct(  # noqa: PLR0912
        self,
        samples: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        modes: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> ttb.tensor:
        """
        Reconstruct or partially reconstruct tensor from ttensor.

        Parameters
        ----------
        samples:
        modes:

        Returns
        -------
        :class:`pyttb.ttensor`
        """
        # Default to sampling full tensor
        full_tensor_sampling = samples is None and modes is None
        if full_tensor_sampling:
            return self.full()

        if modes is not None and samples is None:
            raise ValueError(
                "Samples can be provided without modes, but samples must be provided "
                "with modes."
            )
        assert samples is not None

        if modes is None:
            modes = np.arange(self.ndims)
        elif isinstance(modes, list):
            modes = np.array(modes)
        elif np.isscalar(modes):
            modes = np.array([modes])

        if np.isscalar(samples):
            samples = [np.array([samples])]
        elif not isinstance(samples, list):
            samples = [samples]

        unequal_lengths = len(samples) > 0 and len(samples) != len(modes)
        if unequal_lengths:
            raise ValueError(
                "If samples and modes provided lengths must be equal, but "
                f"samples had length {len(samples)} and modes {len(modes)}"
            )

        full_samples = [np.array([])] * self.ndims
        for sample, mode in zip(samples, modes):
            if np.isscalar(sample):
                full_samples[mode] = np.array([sample])
            else:
                full_samples[mode] = sample

        shape = self.shape
        new_u = []
        for k in range(self.ndims):
            if len(full_samples[k]) == 0:
                # Skip empty samples
                new_u.append(self.factor_matrices[k])
                continue
            if (
                len(full_samples[k].shape) == 2
                and full_samples[k].shape[-1] == shape[k]
            ):
                new_u.append(full_samples[k].dot(self.factor_matrices[k]))
            else:
                new_u.append(self.factor_matrices[k][full_samples[k], :])

        return ttensor(self.core, new_u).full()

    def nvecs(  # noqa: PLR0912
        self, n: int, r: int, flipsign: bool = True
    ) -> np.ndarray:
        """
        Compute the leading mode-n vectors for a ttensor.

        Parameters
        ----------
        n:
            Mode for tensor matricization
        r:
            Number of eigenvalues
        flipsign:
            Make each column's largest element positive if true

        Returns
        -------
        Computed eigenvectors.
        """
        # Compute inner product of all n-1 factors
        V = []
        for factor_idx, factor in enumerate(self.factor_matrices):
            if factor_idx == n:
                V.append(factor)
            else:
                V.append(factor.transpose().dot(factor))
        H = self.core.ttm(V)

        if isinstance(H, ttb.sptensor):
            HnT = H.to_sptenmat(np.array([n]), cdims_cyclic="t").double()
        else:
            HnT = H.full().to_tenmat(cdims=np.array([n])).double()

        G = self.core

        if isinstance(G, ttb.sptensor):
            GnT = G.to_sptenmat(np.array([n]), cdims_cyclic="t").double()
        else:
            GnT = G.full().to_tenmat(cdims=np.array([n])).double()

        # Compute Xn * Xn'
        # Big hack because if RHS is sparse wrong dot product is used
        if sparse.issparse(self.factor_matrices[n]):
            XnT = sparse.coo_matrix.dot(GnT, self.factor_matrices[n].transpose())
        else:
            XnT = GnT.dot(self.factor_matrices[n].transpose())
        if sparse.issparse(XnT):
            Y = sparse.coo_matrix.dot(HnT.transpose(), XnT)
        else:
            Y = HnT.transpose().dot(XnT)

        # TODO: Lifted from tensor, consider common location
        if r < Y.shape[0] - 1:
            w, v = scipy.sparse.linalg.eigsh(Y, r)
            v = v[:, (-np.abs(w)).argsort()]
            v = v[:, :r]
        else:
            logging.debug(
                "Greater than or equal to tensor.shape[n] - 1 eigenvectors requires "
                "cast to dense to solve"
            )
            if sparse.issparse(Y):
                Y = Y.toarray()
            w, v = scipy.linalg.eigh(Y)
            v = v[:, (-np.abs(w)).argsort()]
            v = v[:, :r]

        if flipsign:
            idx = np.argmax(np.abs(v), axis=0)
            for i in range(v.shape[1]):
                if v[idx[i], i] < 0:
                    v[:, i] *= -1
        return v
