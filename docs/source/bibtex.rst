Citing pyttb and Related Work
*****************************

Citing pyttb
============

If you use pyttb in your work, please cite it using the following entry (also
available in `CITATION.bib <https://github.com/sandialabs/pyttb/blob/main/CITATION.bib>`_):

.. code:: bibtex

    @misc{pyttb,
      author = {Dunlavy, Daniel M. and Johnson, Nicholas T. and others},
      month = mar,
      title = {{pyttb: Python Tensor Toolbox, v1.8.6}},
      url = {https://github.com/sandialabs/pyttb},
      year = {2026}
    }

Algorithm references
====================

* ``cp_als``, the tensor classes, and the Tucker methods (``hosvd``, ``tucker_als``) follow the
  Tensor Toolbox for MATLAB: ``TTB_Software``, ``TTB_Dense``, ``TTB_Sparse``.
* ``cp_apr`` is described in ``TTB_CPAPR`` (multiplicative updates) and ``TTB_CPAPRB``
  (Newton-based methods).
* ``gcp_opt`` is described in ``GCP_Hong2020`` and ``GCP_KoldaHong2020``.

Generalized CP references
=========================

.. code:: bibtex

    @article{GCP_Hong2020,
      author = {David Hong and Tamara G. Kolda and Jed A. Duersch},
      title = {Generalized Canonical Polyadic Tensor Decomposition},
      journal = {SIAM Review},
      volume = {62},
      number = {1},
      pages = {133-163},
      year = {2020},
      doi = {10.1137/18M1203626}
    }
    @article{GCP_KoldaHong2020,
      author = {Tamara G. Kolda and David Hong},
      title = {Stochastic Gradients for Large-Scale Tensor Decomposition},
      journal = {SIAM Journal on Mathematics of Data Science},
      volume = {2},
      number = {4},
      pages = {1066-1095},
      year = {2020},
      doi = {10.1137/19m1266265}
    }

Tensor Toolbox for MATLAB references
====================================

.. code:: bibtex

    @misc{TTB_Software,
     author = {Brett W. Bader and Tamara G. Kolda and others},
     title = {MATLAB Tensor Toolbox Version 3.0-dev},
     howpublished = {Available online},
     month = oct,
     year = {2017},
     url = {https://www.tensortoolbox.org}
    }
    @article{TTB_Dense,
      author = {Brett W. Bader and Tamara G. Kolda},
      title = {Algorithm 862: {MATLAB} tensor classes for fast algorithm prototyping},
      journal = {ACM Transactions on Mathematical Software},
      month = dec,
      year = {2006},
      volume = {32},
      number = {4},
      pages = {635-653},
      doi = {10.1145/1186785.1186794}
    }
    @article{TTB_Sparse,
      author = {Brett W. Bader and Tamara G. Kolda},
      title = {Efficient {MATLAB} computations with sparse and factored tensors},
      journal = {SIAM Journal on Scientific Computing},
      month = dec,
      year = {2007},
      volume = {30},
      number = {1},
      pages = {205-231},
      doi = {10.1137/060676489}
    }
    @article{TTB_CPOPT,
      author = {Evrim Acar and Daniel M. Dunlavy and Tamara G. Kolda},
      title = {A Scalable Optimization Approach for Fitting Canonical Tensor Decompositions},
      journal = {Journal of Chemometrics},
      month = feb,
      year = {2011},
      volume = {25},
      number = {2},
      pages = {67-86},
      doi = {10.1002/cem.1335}
    }
    @article{TTB_CPWOPT,
      author = {Evrim Acar and Daniel M. Dunlavy and Tamara G. Kolda and Morten M{\o}rup},
      title = {Scalable Tensor Factorizations for Incomplete Data},
      journal = {Chemometrics and Intelligent Laboratory Systems},
      month = mar,
      year = {2011},
      volume = {106},
      number = {1},
      pages = {41-56},
      doi = {10.1016/j.chemolab.2010.08.004}
    }
    @article{TTB_SSHOPM,
      author = {Tamara G. Kolda and Jackson R. Mayo},
      title = {Shifted Power Method for Computing Tensor Eigenpairs},
      journal = {SIAM Journal on Matrix Analysis and Applications},
      month = oct,
      year = {2011},
      volume = {32},
      number = {4},
      pages = {1095-1124},
      doi = {10.1137/100801482}
    }
    @Article{TTB_EIGGEAP,
      title = {An Adaptive Shifted Power Method for Computing Generalized Tensor Eigenpairs},
      author = {Tamara G. Kolda and Jackson R. Mayo},
      doi = {10.1137/140951758},
      journal = {SIAM Journal on Matrix Analysis and Applications},
      number = {4},
      volume = {35},
      year = {2014},
      month = dec,
      pages = {1563-1581},
    }
    @Article{TTB_CPAPR,
      title = {On Tensors, Sparsity, and Nonnegative Factorizations},
      author = {Eric C. Chi and Tamara G. Kolda},
      doi = {10.1137/110859063},
      journal = {SIAM Journal on Matrix Analysis and Applications},
      number = {4},
      volume = {33},
      year = {2012},
      month = dec,
      pages = {1272-1299},
    }
    @Article{TTB_CPAPRB,
      author = {Samantha Hansen and Todd Plantenga and Tamara G. Kolda},
      title = {Newton-Based Optimization for {Kullback-Leibler} Nonnegative Tensor Factorizations},
      journal = {Optimization Methods and Software},
      volume = {30},
      number = {5},
      pages = {1002-1029},
      month = {April},
      year = {2015},
      doi = {10.1080/10556788.2015.1009977},
    }
    @article{TTB_CPSYM,
      author = {Tamara G. Kolda},
      title = {Numerical Optimization for Symmetric Tensor Decomposition},
      journal = {Mathematical Programming B},
      volume = {151},
      number = {1},
      pages = {225-248},
      month = apr,
      year = {2015},
      doi = {10.1007/s10107-015-0895-0},
    }
    @misc{TTB_CPRALS,
      author = {Casey Battaglino and Grey Ballard and Tamara G. Kolda},
      title = {A Practical Randomized {CP} Tensor Decomposition},
      howpublished = {arXiv:1701.06600},
      month = jan,
      year = {2017},
      eprint = {1701.06600},
      eprintclass = {cs.NA},
    }
    @inproceedings{TTB_MET,
      author = {Tamara G. Kolda and Jimeng Sun},
      title = {Scalable Tensor Decompositions for Multi-aspect Data Mining},
      booktitle = {ICDM 2008: Proceedings of the 8th IEEE International Conference on Data Mining},
      month = dec,
      year = {2008},
      pages = {363-372},
      doi = {10.1109/ICDM.2008.89}
    }
