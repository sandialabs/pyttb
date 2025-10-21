"""Tucker Tensor Implementation."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging
import textwrap
from collections.abc import Sequence
from typing import Literal

import numpy as np
import scipy
from scipy import sparse

import pyttb as ttb
from pyttb import pyttb_utils as ttb_utils
from pyttb.pyttb_utils import OneDArray, parse_one_d, to_memory_order

ALT_CORE_ERROR = "TTensor doesn't support non-tensor cores yet. Only tensor/sptensor."


class ttensor:
    """Class for Tucker tensors (decomposed)."""

    __slots__ = ("core", "factor_matrices")

    def __init__(
        self,
        core: ttb.tensor | ttb.sptensor | None = None,
        factors: Sequence[np.ndarray] | None = None,
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

        >>> core_values = np.ones((2, 2, 2))
        >>> core = ttb.tensor(core_values)
        >>> factors = [np.ones((1, 2))] * len(core_values.shape)
        >>> K0 = ttb.ttensor(core, factors)
        """
        if core is None and factors is None:
            # Empty constructor
            # TODO explore replacing with typing protocol
            self.core: ttb.tensor | ttb.sptensor = ttb.tensor()
            self.factor_matrices: list[np.ndarray] = []
            return

        if core is None or factors is None:
            raise ValueError(
                "For non-empty ttensor both core and factors must be provided"
            )

        if isinstance(core, (ttb.tensor, ttb.sptensor)):
            if not all(
                isinstance(fm, (np.ndarray, sparse.coo_matrix)) for fm in factors
            ):
                raise ValueError(
                    "Factor matrices must be numpy arrays or scipy sparse coo_matrices"
                    f"but received {[type(fm) for fm in factors]}."
                )
            if copy:
                # TODO when generalizing tensor order add order argument to copy
                self.core = core.copy()
                self.factor_matrices = [
                    to_memory_order(fm, self.order, copy=True) for fm in factors
                ]
            else:
                if self.order != core.order:
                    # This isn't possible right now
                    raise ValueError("Core tensor doesn't match Tucker Tensor Order")
                if not all(self._matches_order(factor) for factor in factors):
                    logging.warning(
                        "Selected no copy, but input factor matrices aren't "
                        f"{self.order} ordered so must copy."
                    )
                    factors = [
                        to_memory_order(fm, self.order, copy=True) for fm in factors
                    ]
                self.core = core
                if isinstance(factors, list):
                    self.factor_matrices = factors
                else:
                    logging.warning(
                        "Must provide factor matrices as list to avoid copy"
                    )
                    self.factor_matrices = list(factors)
        else:
            # TODO support any tensor type with supported ops
            raise ValueError(ALT_CORE_ERROR)
        self._validate_ttensor()
        return

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

    def copy(self) -> ttensor:
        """Make a deep copy of a :class:`pyttb.ttensor`.

        Returns
        -------
        Copy of original ttensor.

        Examples
        --------
        >>> core_values = np.ones((2, 2, 2))
        >>> core = ttb.tensor(core_values)
        >>> factors = [np.ones((1, 2))] * len(core_values.shape)
        >>> first = ttb.ttensor(core, factors)
        >>> second = first
        >>> third = second.copy()
        >>> first.factor_matrices[0][0, 0] = 2

        # Item to convert numpy boolean to python boolena for nicer printing

        >>> (first.factor_matrices[0][0, 0] == second.factor_matrices[0][0, 0]).item()
        True
        >>> (first.factor_matrices[0][0, 0] == third.factor_matrices[0][0, 0]).item()
        False
        """
        return ttb.ttensor(self.core, self.factor_matrices, copy=True)

    def __deepcopy__(self, memo):
        """Return deepcopy of class."""
        return self.copy()

    def _validate_ttensor(self):
        """Verify constructed ttensor."""
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
    def shape(self) -> tuple[int, ...]:
        """Shape of the tensor this deconstruction represents."""
        return tuple(factor.shape[0] for factor in self.factor_matrices)

    def __repr__(self):  # pragma: no cover
        """Return string representation of a tucker tensor.

        Returns
        -------
        str
            Contains the core, and factor matrices as strings on different lines.
        """
        display_string = f"TTensor of shape: {self.shape}\n\tCore is a\n"
        display_string += textwrap.indent(str(self.core), "\t\t")
        display_string += "\n"

        for factor_idx, factor in enumerate(self.factor_matrices):
            display_string += f"\tU[{factor_idx}] = \n"
            display_string += textwrap.indent(str(factor), "\t\t")
            display_string += "\n"
        return display_string

    __str__ = __repr__

    def to_tensor(self) -> ttb.tensor:
        """Convert to tensor.

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

    def double(self, immutable: bool = False) -> np.ndarray:
        """Convert ttensor to an array of doubles.

        Parameters
        ----------
        immutable: Whether or not the returned data cam be mutated. May enable
            additional optimizations.
        """
        return self.full().double(immutable)

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
        """Component equality for ttensors.

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
                self.factor_matrices, other.factor_matrices, strict=False
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
        """Unary minus (-) for ttensors.

        Returns
        -------
        :class:`pyttb.ttensor`, copy of tensor
        """
        return ttensor(-self.core, self.factor_matrices)

    def innerprod(
        self, other: ttb.tensor | ttb.sptensor | ttb.ktensor | ttb.ttensor
    ) -> float:
        """Efficient inner product with a ttensor.

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
                self.factor_matrices, other.factor_matrices, strict=False
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
                Z: ttb.tensor | ttb.sptensor = self.full()
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
        """Element wise multiplication (*) for ttensors (only scalars supported).

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
        """Element wise right multiplication (*) for ttensors (only scalars supported).

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
        vector: Sequence[np.ndarray] | np.ndarray,
        dims: OneDArray | None = None,
        exclude_dims: OneDArray | None = None,
    ) -> float | ttensor:
        """TTensor times vector.

        Parameters
        ----------
        vector:
            Vector to multiply by.
        dims:
            Dimensions to multiply in.
        exclude_dims:
            Alternative multiply by all dimensions but these.
        """
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

    def mttkrp(
        self, U: ttb.ktensor | Sequence[np.ndarray], n: int | np.integer
    ) -> np.ndarray:
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

        W = [np.empty((), order=self.order)] * self.ndims
        if isinstance(U, ttb.ktensor):
            U = U.factor_matrices
        for i in range(self.ndims):
            if i == n:
                continue
            W[i] = self.factor_matrices[i].transpose().dot(U[i])

        Y = self.core.mttkrp(W, n)

        # Find each column of answer by multiplying by weights
        return to_memory_order(self.factor_matrices[n].dot(Y), self.order)

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

    def permute(self, order: OneDArray) -> ttensor:
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
        order = parse_one_d(order)
        if not np.array_equal(np.arange(0, self.ndims), np.sort(order)):
            raise ValueError("Invalid permutation")
        new_core = self.core.permute(order)
        new_u = [self.factor_matrices[idx] for idx in order]
        return ttensor(new_core, new_u)

    def ttm(
        self,
        matrix: np.ndarray | Sequence[np.ndarray],
        dims: float | np.ndarray | None = None,
        exclude_dims: int | np.ndarray | None = None,
        transpose: bool = False,
    ) -> ttensor:
        """Tensor times matrix for ttensor.

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
            dims = np.array(dims, order=self.order)
        elif isinstance(dims, (float, int, np.generic)):
            dims = np.array([dims], order=self.order)

        if isinstance(exclude_dims, (float, int)):
            exclude_dims = np.array([exclude_dims], order=self.order)

        if not isinstance(matrix, Sequence):
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
        samples: np.ndarray | Sequence[np.ndarray] | None = None,
        modes: np.ndarray | Sequence[np.ndarray] | None = None,
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
        elif isinstance(modes, Sequence):
            modes = np.array(modes, order=self.order)
        elif np.isscalar(modes):
            modes = np.array([modes], order=self.order)

        if np.isscalar(samples):
            samples = [np.array([samples], order=self.order)]
        elif not isinstance(samples, Sequence):
            samples = [samples]

        unequal_lengths = len(samples) > 0 and len(samples) != len(modes)
        if unequal_lengths:
            raise ValueError(
                "If samples and modes provided lengths must be equal, but "
                f"samples had length {len(samples)} and modes {len(modes)}"
            )

        full_samples = [np.array([], order=self.order)] * self.ndims
        for sample, mode in zip(samples, modes, strict=False):
            if np.isscalar(sample):
                full_samples[mode] = np.array([sample], order=self.order)
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
            HnT = H.to_sptenmat(
                np.array([n], order=self.order), cdims_cyclic="t"
            ).double()
        else:
            HnT = H.full().to_tenmat(cdims=np.array([n], order=self.order)).double()

        G = self.core

        if isinstance(G, ttb.sptensor):
            GnT = G.to_sptenmat(
                np.array([n], order=self.order), cdims_cyclic="t"
            ).double()
        else:
            GnT = G.full().to_tenmat(cdims=np.array([n], order=self.order)).double()

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


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
