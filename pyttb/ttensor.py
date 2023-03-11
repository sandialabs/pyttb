# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import textwrap
import warnings

import numpy as np
import scipy

from pyttb import ktensor
from pyttb import pyttb_utils as ttb_utils
from pyttb import sptenmat, sptensor, tenmat, tensor

ALT_CORE_ERROR = "TTensor doesn't support non-tensor cores yet"


class ttensor(object):
    """
    TTENSOR Class for Tucker tensors (decomposed).

    """

    def __init__(self):
        """
        Create an empty decomposed tucker tensor

        Returns
        -------
        :class:`pyttb.ttensor`
        """
        # Empty constructor
        self.core = tensor()
        self.u = []

    @classmethod
    def from_data(cls, core, factors):
        """
        Construct an ttensor from fully defined core tensor and factor matrices.

        Parameters
        ----------
        core: :class: `ttb.tensor`
        factors: :class:`list(numpy.ndarray)`

        Returns
        -------
        :class:`pyttb.ttensor`

        Examples
        --------
        Import required modules:

        >>> import pyttb as ttb
        >>> import numpy as np

        Set up input data
        # Create ttensor with explicit data description

        >>> core_values = np.ones((2,2,2))
        >>> core = ttb.tensor.from_data(core_values)
        >>> factors = [np.ones((1,2))] * len(core_values.shape)
        >>> K0 = ttb.ttensor.from_data(core, factors)
        """
        ttensorInstance = ttensor()
        if isinstance(core, tensor):
            ttensorInstance.core = tensor.from_data(core.data, core.shape)
            ttensorInstance.u = factors.copy()
        else:
            # TODO support any tensor type with supported ops
            raise ValueError("TTENSOR doesn't yet support generic cores, only tensor")
        ttensorInstance._validate_ttensor()
        return ttensorInstance

    @classmethod
    def from_tensor_type(cls, source):
        """
        Converts other tensor types into a ttensor

        Parameters
        ----------
        source: :class:`pyttb.ttensor`

        Returns
        -------
        :class:`pyttb.ttensor`
        """
        # Copy Constructor
        if isinstance(source, ttensor):
            return cls.from_data(source.core, source.u)

    def _validate_ttensor(self):
        """
        Verifies the validity of constructed ttensor

        Returns
        -------
        """
        # Confirm all factors are matrices
        for factor_idx, factor in enumerate(self.u):
            if not isinstance(factor, np.ndarray):
                raise ValueError(
                    f"Factor matrices must be numpy arrays but factor {factor_idx} was {type(factor)}"
                )
            if len(factor.shape) != 2:
                raise ValueError(
                    f"Factor matrix {factor_idx} has shape {factor.shape} and is not a matrix!"
                )

        # Verify size consistency
        core_order = len(self.core.shape)
        num_matrices = len(self.u)
        if core_order != num_matrices:
            raise ValueError(
                f"CORE has order {core_order} but there are {num_matrices} factors"
            )
        for factor_idx, factor in enumerate(self.u):
            if factor.shape[-1] != self.core.shape[factor_idx]:
                raise ValueError(
                    f"Factor matrix {factor_idx} does not have {self.core.shape[factor_idx]} columns"
                )

    @property
    def shape(self):
        """
        Shape of the tensor this deconstruction represents.

        Returns
        -------
        tuple(int)
        """
        return tuple(factor.shape[0] for factor in self.u)

    def __repr__(self):  # pragma: no cover
        """
        String representation of a tucker tensor.

        Returns
        -------
        str
            Contains the core, and factor matrices as strings on different lines.
        """
        display_string = f"Tensor of shape: {self.shape}\n" f"\tCore is a "
        display_string += textwrap.indent(str(self.core), "\t")

        for factor_idx, factor in enumerate(self.u):
            display_string += f"\tU[{factor_idx}] = \n"
            display_string += textwrap.indent(str(factor), "\t\t")
            display_string += "\n"
        return display_string

    __str__ = __repr__

    def full(self):
        """
        Convert a ttensor to a (dense) tensor.

        Returns
        -------
        :class:`pyttb.tensor`
        """
        recomposed_tensor = self.core.ttm(self.u)

        # There is a small chance tensor could be sparse so ensure we cast that to dense.
        if not isinstance(recomposed_tensor, tensor):
            raise ValueError(ALT_CORE_ERROR)
        return recomposed_tensor

    def double(self):
        """
        Convert ttensor to an array of doubles

        Returns
        -------
        :class:`numpy.ndarray`
            copy of tensor data
        """
        return self.full().double()

    @property
    def ndims(self):
        """
        Number of dimensions of a ttensor.

        Returns
        -------
        int
            Number of dimensions of ttensor
        """
        return len(self.u)

    def isequal(self, other):
        """
        Component equality for ttensors

        Parameters
        ----------
        other: :class:`pyttb.ttensor`

        Returns
        -------
        bool: True if ttensors decompositions are identical, false otherwise
        """
        if not isinstance(other, ttensor):
            return False
        if self.ndims != other.ndims:
            return False
        return self.core.isequal(other.core) and all(
            np.array_equal(this_factor, other_factor)
            for this_factor, other_factor in zip(self.u, other.u)
        )

    def __pos__(self):
        """
        Unary plus (+) for ttensors. Does nothing.

        Returns
        -------
        :class:`pyttb.ttensor`, copy of tensor
        """

        return ttensor.from_tensor_type(self)

    def __neg__(self):
        """
        Unary minus (-) for ttensors

        Returns
        -------
        :class:`pyttb.ttensor`, copy of tensor
        """

        return ttensor.from_data(-self.core, self.u)

    def innerprod(self, other):
        """
        Efficient inner product with a ttensor

        Parameters
        ----------
        other: :class:`pyttb.tensor`, :class:`pyttb.sptensor`, :class:`pyttb.ktensor`,
        :class:`pyttb.ttensor`

        Returns
        -------
        float
        """
        if isinstance(other, ttensor):
            if self.shape != other.shape:
                raise ValueError(
                    "ttensors must have same shape to perform an innerproduct, but this ttensor "
                    f"has shape {self.shape} and the other has {other.shape}"
                )
            if np.prod(self.core.shape) > np.prod(other.core.shape):
                # Reverse arguments so the ttensor with the smaller core comes first.
                return other.innerprod(self)
            W = []
            for this_factor, other_factor in zip(self.u, other.u):
                W.append(this_factor.transpose().dot(other_factor))
            J = other.core.ttm(W)
            return self.core.innerprod(J)
        elif isinstance(other, (tensor, sptensor)):
            if self.shape != other.shape:
                raise ValueError(
                    "ttensors must have same shape to perform an innerproduct, but this ttensor "
                    f"has shape {self.shape} and the other has {other.shape}"
                )
            if np.prod(self.shape) < np.prod(self.core.shape):
                Z = self.full()
                return Z.innerprod(other)
            Z = other.ttm(self.u, transpose=True)
            return Z.innerprod(self.core)
        elif isinstance(other, ktensor):
            # Call ktensor implementation
            # TODO needs ttensor ttv
            return other.innerprod(self)
        else:
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
            return ttensor.from_data(self.core * other, self.u)
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

    def ttv(self, vector, dims=None):
        """
        TTensor times vector

        Parameters
        ----------
        vector: :class:`Numpy.ndarray`, list[:class:`Numpy.ndarray`]
        dims: :class:`Numpy.ndarray`, int
        """
        if dims is None:
            dims = np.array([])
        # TODO make helper function to check scalar since re-used many places
        elif isinstance(dims, (float, int)):
            dims = np.array([dims])

        # Check that vector is a list of vectors, if not place single vector as element in list
        if len(vector) > 0 and isinstance(vector[0], (int, float, np.int_, np.float_)):
            return self.ttv(np.array([vector]), dims)

        # Get sorted dims and index for multiplicands
        dims, vidx = ttb_utils.tt_dimscheck(dims, self.ndims, len(vector))

        # Check that each multiplicand is the right size.
        for i in range(dims.size):
            if vector[vidx[i]].shape != (self.shape[dims[i]],):
                raise ValueError("Multiplicand is wrong size")

        # Get remaining dimensions when we're done
        remdims = np.setdiff1d(np.arange(0, self.ndims), dims)

        # Create W to multiply with core, only populated remaining dims
        W = [None] * len(dims)
        for i in range(dims.size):
            dim_idx = dims[i]
            W[dim_idx] = self.u[dim_idx].transpose().dot(vector[vidx[i]])

        # Create new core
        newcore = self.core.ttv(W, dims)

        # Create final result
        if remdims.size == 0:
            return newcore
        else:
            return ttensor.from_data(newcore, [self.u[dim] for dim in remdims])

    def mttkrp(self, U, n):
        """
        Matricized tensor times Khatri-Rao product for ttensors.

        Parameters
        ----------
        U: array of matrices or ktensor
        n: multiplies by all modes except n

        Returns
        -------
        :class:`numpy.ndarray`
        """
        # NOTE: MATLAB version calculates an unused R here

        W = [None] * self.ndims
        for i in range(0, self.ndims):
            if i == n:
                continue
            W[i] = self.u[i].transpose().dot(U[i])

        Y = self.core.mttkrp(W, n)

        # Find each column of answer by multiplying by weights
        return self.u[n].dot(Y)

    def norm(self):
        """
        Compute the norm of a ttensor.
        Returns
        -------
        norm: float, Frobenius norm of Tensor
        """
        if np.prod(self.shape) > np.prod(self.core.shape):
            V = []
            for factor in self.u:
                V.append(factor.transpose().dot(factor))
            Y = self.core.ttm(V)
            tmp = Y.innerprod(self.core)
            return np.sqrt(tmp)
        else:
            return self.full().norm()

    def permute(self, order):
        """
        Permute dimensions for a ttensor

        Parameters
        ----------
        order: :class:`Numpy.ndarray`

        Returns
        -------
        :class:`pyttb.ttensor`
        """
        if not np.array_equal(np.arange(0, self.ndims), np.sort(order)):
            raise ValueError("Invalid permutation")
        new_core = self.core.permute(order)
        new_u = [self.u[idx] for idx in order]
        return ttensor.from_data(new_core, new_u)

    def ttm(self, matrix, dims=None, transpose=False):
        """
        Tensor times matrix for ttensor

        Parameters
        ----------
        matrix: :class:`Numpy.ndarray`, list[:class:`Numpy.ndarray`]
        dims: :class:`Numpy.ndarray`, int
        transpose: bool
        """
        if dims is None:
            dims = np.arange(self.ndims)
        elif isinstance(dims, list):
            dims = np.array(dims)
        elif np.isscalar(dims):
            if dims < 0:
                raise ValueError("Negative dims is currently unsupported, see #62")
            dims = np.array([dims])

        if not isinstance(matrix, list):
            return self.ttm([matrix], dims, transpose)

        # Check that the dimensions are valid
        dims, vidx = ttb_utils.tt_dimscheck(dims, self.ndims, len(matrix))

        # Determine correct size index
        size_idx = int(not transpose)

        # Check that each multiplicand is the right size.
        for i in range(len(dims)):
            if matrix[vidx[i]].shape[size_idx] != self.shape[dims[i]]:
                raise ValueError(f"Multiplicand {i} is wrong size")

        # Do the actual multiplications in the specified modes.
        new_u = self.u.copy()
        for i in range(len(dims)):
            if transpose:
                new_u[dims[i]] = matrix[vidx[i]].transpose().dot(new_u[dims[i]])
            else:
                new_u[dims[i]] = matrix[vidx[i]].dot(new_u[dims[i]])

        return ttensor.from_data(self.core, new_u)

    def reconstruct(self, samples=None, modes=None):
        """
        Reconstruct or partially reconstruct tensor from ttensor.

        Parameters
        ----------
        samples: :class:`Numpy.ndarray`, list[:class:`Numpy.ndarray`]
        modes: :class:`Numpy.ndarray`, list[:class:`Numpy.ndarray`]

        Returns
        -------
        :class:`pyttb.ttensor`
        """
        # Default to sampling full tensor
        full_tensor_sampling = samples is None and modes is None
        if full_tensor_sampling:
            return self.full()

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

        unequal_lengths = len(samples) != len(modes)
        if unequal_lengths:
            raise ValueError(
                "If samples and modes provides lengths must be equal, but "
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
                new_u.append(self.u[k])
                continue
            elif (
                len(full_samples[k].shape) == 2
                and full_samples[k].shape[-1] == shape[k]
            ):
                new_u.append(full_samples[k].dot(self.u[k]))
            else:
                new_u.append(self.u[k][full_samples[k], :])

        return ttensor.from_data(self.core, new_u).full()

    def nvecs(self, n, r, flipsign=True):
        """
        Compute the leading mode-n vectors for a ttensor.

        Parameters
        ----------
        n: mode for tensor matricization
        r: number of eigenvalues
        flipsign: Make each column's largest element positive if true

        Returns
        -------
        :class:`numpy.ndarray`
        """
        # Compute inner product of all n-1 factors
        V = []
        for factor_idx, factor in enumerate(self.u):
            if factor_idx == n:
                V.append(factor)
            else:
                V.append(factor.transpose().dot(factor))
        H = self.core.ttm(V)

        if isinstance(H, sptensor):
            raise NotImplementedError(ALT_CORE_ERROR)
        else:
            HnT = tenmat.from_tensor_type(H.full(), cdims=np.array([n])).double()

        G = self.core

        if isinstance(G, sptensor):
            raise NotImplementedError(ALT_CORE_ERROR)
        else:
            GnT = tenmat.from_tensor_type(G.full(), cdims=np.array([n])).double()

        # Compute Xn * Xn'
        Y = HnT.transpose().dot(GnT.dot(self.u[n].transpose()))

        # TODO: Lifted from tensor, consider common location
        if r < Y.shape[0] - 1:
            w, v = scipy.sparse.linalg.eigsh(Y, r)
            v = v[:, (-np.abs(w)).argsort()]
            v = v[:, :r]
        else:
            warnings.warn(
                "Greater than or equal to tensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
            )
            w, v = scipy.linalg.eigh(Y)
            v = v[:, (-np.abs(w)).argsort()]
            v = v[:, :r]

        if flipsign:
            idx = np.argmax(np.abs(v), axis=0)
            for i in range(v.shape[1]):
                if v[idx[i], i] < 0:
                    v[:, i] *= -1
        return v
