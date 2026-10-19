Developer Guide
===============

This page collects the conventions used in the pyttb code and documentation.
For the contribution workflow (issues, forks, pull requests) see
`CONTRIBUTING <https://github.com/sandialabs/pyttb/blob/main/CONTRIBUTING.md>`_.

Documentation conventions
-------------------------

Method or function docstring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Docstrings use numpy style, and type hints carry the types
(``napoleon_use_param=False``). Sections appear in this order:

1. Summary line ending with a period.
2. Optional extended description.
3. ``Parameters``
4. ``Returns``
5. ``Raises`` (when the function raises)
6. ``See Also`` (optional)
7. ``Notes`` (optional)
8. ``Examples``

Rules for each section:

* **Parameters:** one ``name:`` line per parameter, in signature order, with an indented
  description. Write variadic parameters as ``*inputs:`` / ``**kwargs:``. No types in this
  section.
* **Returns, single value:** one description sentence on its own line, with no name, no type
  and no lone ``:``. Example: ``Dense tensor with the same entries.``
* **Returns, tuple:** one ``name:`` entry per element, with an indented description. Use the
  same names as the Examples. For algorithms these are ``M:``, ``M_init:`` and ``M_info:``.
  Describe a dict return with a bullet list of its keys.
* **Raises:** ``ValueError`` on one line, then an indented line starting "If ...".
* **Examples:** a prose sentence ending in ``::``, then an indented ``>>>`` block, following
  the rules in `Doctest rules`_.
* **Properties:** a summary, plus Examples where the template classes have them. No
  Parameters or Returns.
* **Exempt from Examples:** ``order``, ``__deepcopy__``, gcp objective handles, and optimizer
  hooks (``update_step``, ``set_failed_epoch``, ``solve``).

Class docstring
^^^^^^^^^^^^^^^

Keep the current template, with constructor arguments in the class docstring and no
``__init__`` docstring. The template is in ``pyttb/ktensor.py``, class docstring, and its
sections are:

1. Summary.
2. ``Parameters``.
3. ``Examples``, one per constructor form.
4. ``Notes``, listing the other ways to create an instance.
5. ``**Tutorial**``, linking ``:doc:`/tutorial/<name>```.
6. ``**Attributes and Methods**``.

``__slots__`` is a dict mapping each attribute to its description.

Algorithm docstring
^^^^^^^^^^^^^^^^^^^

Use the method template, plus:

* a ``Returns`` section listing ``M``, ``M_init`` and ``M_info`` (with its dict keys).
  ``hosvd`` returns only ``M``;
* a ``Raises`` section;
* a ``See Also`` section naming the related algorithms;
* a final bold ``**Tutorial**`` line pointing to ``/tutorial/algorithm_<name>``.

.rst API page
^^^^^^^^^^^^^

Copy ``docs/source/ktensor.rst``:

1. Title: ``<Name> (:class:`pyttb.<x>`)`` with a ``===`` underline.
2. The sentence "For *all* examples in this document, the following module imports are
   assumed::" followed by the two imports.
3. ``autoclass`` with ``:members:``, ``:special-members:`` and
   ``:exclude-members: __init__, __dict__, __weakref__, __deepcopy__``.

Tutorials
^^^^^^^^^

* **H1 title:** must equal the link text in ``tutorials.rst``.
* **Required opening:** the H1, the copyright block, a one-paragraph intro, an "API reference"
  line linking the ``.rst`` page (e.g. ``[API reference](../ktensor.rst)``), then the imports
  cell.
* **Outlines:**

  * Class tutorials: ``## Creating instances``, ``## Key properties``,
    ``## Converting to other types``, ``## Core methods``, ``## Operators``,
    ``## Additional examples``. Omit groups that would be empty.
  * Algorithm tutorials: ``## Problem setup``, ``## Basic usage``, ``## Main options``,
    ``## Interpreting the output``, ``## Worked example``.
  * Utility tutorials: ``## Overview``, then one ``##`` section per function family, then
    ``## Summary`` with links to the API pages.

* **Synthetic data:** describe how the data is generated, not a statistical model, except
  for ``cp_apr`` (Poisson).

Variable names (algorithms)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same convention applies to algorithm docstring Examples and to algorithm tutorials.
There is no separate tutorial convention.

* **Decomposition outputs** ("info" is kept for consistency with Tensor Toolbox for MATLAB):

  * Unlabeled run: ``M, M_init, M_info = ttb.<alg>(...)``.
  * Labeled run: ``M_<label>, M_<label>_init, M_<label>_info``, where the label is a number
    or a short tag. Examples: ``M_1``, ``M_1_init``, ``M_1_info``; ``M_lbfgs``,
    ``M_lbfgs_init``, ``M_lbfgs_info``; ``M_nvecs``; ``M_ztp``.
  * ``M_<label>`` is the decomposition, ``M_<label>_init`` its initial guess, and
    ``M_<label>_info`` its output information.
  * ``hosvd`` returns only the model: ``M = ttb.hosvd(...)`` (or ``M_<label>``).
  * Use ``_`` for outputs you don't need. Never assign the whole tuple to one name.

* **Model that generates the data:** ``M_true`` (a ``ktensor`` or ``ttensor``). Use
  ``M_true_<label>`` when a docstring or notebook needs a second one. Its building blocks
  are ``G`` (core) and ``U`` or ``factors`` (factor matrices).
* **Data tensors:** ``X``, with ``X_<label>`` for variants such as ``X_noisy``, ``X_sigma``
  and ``X_large``.
* **Score against the true model:** ``score_<model name>``, e.g. ``score_M_1``.

Doctest rules
-------------

Doctests run through ``addopts = "--doctest-modules pyttb"`` in ``pyproject.toml``, so a plain
``pytest`` runs every docstring example. The doctest namespace (see ``conftest.py``) already
provides ``np`` and ``ttb``, so examples do not need to import them.

* Use tensors with 3 or more dimensions where the method allows it.
* Seed randomness with ``np.random.seed(n)``.
* Show floats as ``round(float(x), k)`` or with an f-string format. Avoid long raw floats.
* Use ``# doctest: +NORMALIZE_WHITESPACE`` for multi-line reprs, and no tabs.
* Paste real output: run the statements in Python and copy the result; never type it from
  memory. Replace tabs in pasted output with 4 spaces and strip trailing spaces.
* Never delete a failing example to get a pass. Fix the example.

The script ``docs/tools/doc_audit.py`` checks the rendered docstrings against these
conventions::

    python docs/tools/doc_audit.py --summary          # counts per category
    python docs/tools/doc_audit.py pyttb.tensor       # findings for one module prefix

It exits non-zero while any finding remains.

Adding a class, algorithm, or utility
-------------------------------------

1. Add the module under ``pyttb/``.
2. Import it in ``pyttb/__init__.py`` and add it to ``__all__`` (use string literals).
3. Add the ``.rst`` API page, following the template above.
4. Add the page to the matching toctree (``tensor_classes.rst``, ``algorithms.rst``,
   ``reference.rst``, ...).
5. Add a tutorial notebook under ``docs/source/tutorial/`` and a link in ``tutorials.rst``.
6. Add the MATLAB mapping row to the matching page under ``docs/source/matlab/``.
7. Add tests in ``tests/test_<module>.py``.
8. Run ``python docs/tools/doc_audit.py`` and fix any findings.

Building docs and tutorials
---------------------------

From the project root, with the ``doc`` extras installed
(``python -m pip install -e ".[dev,doc]"``)::

    sphinx-build ./docs/source ./docs/build

For the CI version, which is more strict::

    sphinx-build ./docs/source ./docs/build -E -W --keep-going

If not on Windows, optionally add ``-j auto`` for parallelization.

Tutorials are Jupyter notebooks in ``docs/source/tutorial``. The filename is lower case and
prefixed with the type of tutorial (``class_``, ``algorithm_``, ``utility_``). Add the notebook
to ``tutorials.rst``, rebuild the docs, and review locally. Notebooks are rendered with
`myst-nb <https://myst-nb.readthedocs.io/>`_ and executed during the build; each cell must run in
under 60 seconds.

Edit notebooks programmatically with ``nbformat`` rather than editing the JSON by hand. Clear
outputs and volatile metadata before committing (see ``nbstripout`` in the
``.pre-commit-config.yaml``)::

    nbstripout --extra-keys="metadata.language_info metadata.vscode metadata.kernelspec cell.metadata.vscode" --drop-empty-cells docs/source/tutorial/*.ipynb

Test layout
-----------

* ``tests/`` mirrors the modules: ``tests/test_<module>.py`` tests ``pyttb/<module>.py``.
* ``tests/data`` holds the fixtures for the import/export tests.
* Doctests run through ``addopts`` (see `Doctest rules`_).
* Run just the doctests with ``pytest``, the functional tests with ``pytest tests``.

Release checklist
-----------------

1. Bump ``__version__`` in ``pyttb/__init__.py``.
2. Update ``CITATION.bib`` (version, month, year).
3. Keep the pyttb entry in ``bibtex.rst`` in sync.
4. The maintainer writes the CHANGELOG.

Method ordering
---------------

See the order of methods in classes in
`CONTRIBUTING <https://github.com/sandialabs/pyttb/blob/main/CONTRIBUTING.md>`_.
