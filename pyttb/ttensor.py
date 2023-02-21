# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from pyttb import tensor
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
