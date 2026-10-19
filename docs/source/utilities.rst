Utilities
=========

For *all* examples in this document, the following module imports are assumed::

    >>> import pyttb as ttb
    >>> import numpy as np

.. toctree::
    :maxdepth: 1

    create_problem.rst
    io.rst

Khatri-Rao Product (khatrirao)
------------------------------

.. autofunction:: pyttb.khatrirao.khatrirao

Warnings (:func:`pyttb.ignore_warnings`)
----------------------------------------

.. autofunction:: pyttb.ignore_warnings

MATLAB Support
--------------

Helpers for users transitioning from MATLAB, such as
:func:`pyttb.matlab.matlab_support.matlab_print`, are described in
:doc:`for_matlab_users`.

Tutorials
---------

* :doc:`Creating Tensors <tutorial/utility_tensor_creation>`
* :doc:`Tensor Multiplication <tutorial/utility_multiplication>` (including ``khatrirao``)
* :doc:`Collapsing and Scaling Tensors <tutorial/utility_collapse_scale>`
* :doc:`Reading and Writing Tensors (import_data, export_data) <tutorial/utility_io>`
* :doc:`Creating Test Problems (create_problem) <tutorial/utility_test_problem>`
