# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from pyttb import (
    ktensor,
    tensor,
    sptensor,
)
import numpy as np
import textwrap

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
        # Create sptensor with explicit data description

        >>> core_values = np.ones((2,2,2))
        >>> core = ttb.tensor.from_data(core_values)
        >>> factors = [np.ones((1,2))] * len(core_values)
        >>> K0 = ttb.ttensor.from_data(core, factors)
        """
        ttensorInstance = ttensor()
        if isinstance(core, tensor):
            ttensorInstance.core = tensor.from_data(core.data, core.shape)
            ttensorInstance.u = factors
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
            if len(factor.shape) != 2:
                raise ValueError(f"Factor matrix {factor_idx} has shape {factor.shape} and is not a matrix!")

        # Verify size consistency
        core_order = len(self.core.shape)
        num_matrices = len(self.u)
        if core_order != num_matrices:
            raise ValueError(f"CORE has order {core_order} but there are {num_matrices}")
        for factor_idx, factor in enumerate(self.u):
            if factor.shape[-1] != self.core.shape[factor_idx]:
                raise ValueError(f"Factor matrix {factor_idx} does not have {self.core.shape[factor_idx]} columns")

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
        display_string = (
            f"Tensor of shape: {self.shape}\n"
            f"\tCore is a "
        )
        display_string += textwrap.indent(str(self.core), '\t')

        for factor_idx, factor in enumerate(self.u):
            display_string += f"\tU[{factor_idx}] = \n"
            display_string += textwrap.indent(str(factor), '\t\t')
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
            recomposed_tensor = tensor(recomposed_tensor)
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
                np.array_equal(this_factor, other_factor) for this_factor, other_factor in zip(self.u, other.u)
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
            for (this_factor, other_factor) in zip(self.u, other.u):
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
            raise ValueError(f"Inner product between ttensor and {type(other)} is not supported")

    def __mul__(self, other):
        """
        Element wise multiplication (*) for ttensors (only scalars supported)

        Parameters
        ----------
        float, int

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
        float, int

        Returns
        -------
        :class:`pyttb.ttensor`
        """
        if isinstance(other, (float, int, np.number)):
            return self.__mul__(other)
        raise ValueError("This object cannot be multiplied by ttensor")
