.. Python Tensor Toolbox documentation master file

pyttb: Python Tensor Toolbox
****************************
Tensors (also known as multidimensional arrays or N-way arrays) are used
in a variety of applications ranging from chemometrics to network
analysis.

-  This is open source software. Please see `LICENSE`_ for the
   terms of the license (2-clause BSD).
-  For more information or for feedback on this project, please `contact us`_.

.. _`LICENSE`: ../../../LICENSE
.. _contact us: #contact

Functionality
==============
pyttb provides the following classes and functions
for manipulating dense, sparse, and structured tensors, along with
algorithms for computing low-rank tensor models.

- `Tensor Classes`_ 

   pyttb supports multiple tensor types, including
   dense and sparse, as well as specially structured tensors, such as
   the Krusal format (stored as factor matrices).

- `Algorithms`_

   CP methods such as alternating least squares, direct optimization,
   and weighted optimization (for missing data).  Also alternative
   decompositions such as Poisson Tensor Factorization via alternating
   Poisson regression.

.. _Tensor Classes: tensor_classes.html
.. _Algorithms: algorithms.html


Tutorials
=========

.. toctree::
   :maxdepth: 1

   tutorials


Python API
================

.. toctree::
   :maxdepth: 2

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
