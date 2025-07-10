.. Python Tensor Toolbox documentation master file

pyttb: Python Tensor Toolbox
****************************
Tensors (also known as multidimensional arrays or N-way arrays) are used
in a variety of applications ranging from chemometrics to network
analysis. 

-  This Python package is an adaptation of the open source `Tensor Toolbox for MATLAB <https://www.tensortoolbox.org>`_.
-  This is open source software. Please see `LICENSE <https://github.com/sandialabs/pyttb/blob/main/LICENSE>`_ for the
   terms of the license (2-clause BSD).
-  The pyttb code repository is hosted at `github <https://github.com/sandialabs/pyttb>`_. 
-  Contributions are welcome; see `CONTRIBUTING <https://github.com/sandialabs/pyttb/blob/main/CONTRIBUTING.md>`_.
-  Reports of bugs and feature requests can be `submitted on github <https://github.com/sandialabs/pyttb/issues>`_.
-  For more information or for feedback on this project, please `contact us`_.

.. _`LICENSE`: ../../../LICENSE
.. _contact us: #contact


Installing
==========

* Via pypi
   -  Install the latest release from pypi (``pip install pyttb`` or ``python -m pip install pyttb``)
* From source
   -  Clone the repository from `github <https://github.com/sandialabs/pyttb>`_.
   -  Install the package with ``pip install .`` from the pyttb root directory. 
   -  Note: Use ``pip install -e ".[dev,doc]"`` if you are planning development.

Functionality
==============
pyttb provides the following classes and functions
for manipulating dense, sparse, and structured tensors, along with
algorithms for computing low-rank tensor models.

- `Tensor Classes`_ 

   pyttb supports multiple tensor types, including
   dense and sparse, as well as specially structured tensors, such as
   the Kruskal format (stored as factor matrices).

- `Algorithms`_

   CP methods such as alternating least squares, direct optimization,
   and weighted optimization (for missing data).  Also alternative
   decompositions such as Poisson Tensor Factorization via alternating
   Poisson regression.

.. _Tensor Classes: tensor_classes.html
.. _Algorithms: algorithms.html


Getting Started
===============
For historical reasons we use Fortran memory layouts, where numpy by default uses C.
This is relevant for indexing. In the future we hope to extend support for both.

.. code-block:: python

    >>> import numpy as np
    >>> c_order = np.arange(8).reshape((2,2,2))
    >>> f_order = np.arange(8).reshape((2,2,2), order="F")
    >>> print(c_order[0,1,1])
    3
    >>> print(f_order[0,1,1])
    6

.. toctree::
   :maxdepth: 1

   getting_started.rst

Python Reference
================

.. toctree::
   :maxdepth: 3

   reference.rst

How to Cite
==============
Please see `references`_ for how to cite a variety of algorithms implemented in this project.

.. _references: bibtex.html

.. toctree::
   :maxdepth: 2

   bibtex.rst

Contact
================

Please email dmdunla@sandia.gov with any questions about pyttb
that cannot be resolved via issue reporting. Stories of its usefulness
are especially welcome. We will try to respond to every email may not
always be successful due to the volume of emails.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
