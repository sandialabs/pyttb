# pyttb User Guide and Rubric

The pyttb package is a powerful toolset for working with tensors in Python, designed to cater to a wide range of users, from beginners to advanced. This user guide and rubric will assist new users in understanding the capabilities of the pyttb package and how it can meet their specific needs.

## Installation Guide

For detailed instructions on how to install and set up pyttb, please refer to our [CONTRIBUTING.md](https://github.com/sandialabs/pyttb/blob/main/CONTRIBUTING.md) guide.


## User Categories

Users can generally be classified into three main categories:

1. **Novice Users:** Users with no or little previous experience with Python, Matlab, or tensor toolbox. These users might require detailed explanations and step-by-step guidance.

2. **Intermediate Users:** Users with some level of familiarity with Python and Matlab, and a basic understanding of tensor operations. They are likely to be seeking more advanced features and functionalities.

3. **Expert Users:** Users with a strong understanding of Python, Matlab, and tensor operations. They are probably looking for advanced features, custom options, or performance-optimizing/specific functions. 

## Features and Matching User Levels

This section provides an overview of pyttb's key features, a description of each, the user level they are best suited to, and some potential use cases.

| Feature Name | Description          | User Level                                 | Potential Use Cases        |
|--------------|----------------------|--------------------------------------------|----------------------------|
| Tensor Types | Supports multiple tensor types, including dense, sparse, symmetric, and specially structured tensors | Novice/Intermediate/Expert | Tensor analysis and manipulation |
| CP Decompositions | Implements CP methods such as alternating least squares, direct optimization, and weighted optimization | Intermediate/Expert | Tensor factorization and data analysis |
| Tucker Decomposition | Includes Tucker methods such as HOSVD, ST-HOSVD, and HOOI | Intermediate/Expert | Tensor decomposition and data reduction |
| Eigenproblems | Provides methods to solve tensor eigenproblems | Expert | Advanced tensor analysis and machine learning |
| Tensor Operations | Supports creating test problems, tensor multiplication, collapsing and scaling tensors | Novice/Intermediate/Expert | Preprocessing, tensor manipulation, and creating test scenarios |
| Optimization Methods | Offers standardized wrappers for various optimization methods | Intermediate/Expert | Optimization tasks and performance enhancement |

## Troubleshooting / FAQ

Refer to our [Issues tab](https://github.com/sandialabs/pyttb/issues) for known issues and solutions.


## Sample Code / Tutorials

 The [tutorial](https://github.com/sandialabs/pyttb/tree/main/docs/source/tutorial) section contains step-by-step tutorials with Jupyter notebooks demonstrating various pyttb features.



## Getting Started (for Novice Users)

Those new to the tensor toolbox or Python/Matlab should check the [tutorial](#sample-code--tutorials).

## Advanced Usage (for Intermediate and Expert Users)

`TODO` ... Advanced functionalities available in pyttb. 


## Special Topics (for Expert Users)

`TODO` ... More specific, in-depth topics for pyttb research.

Reference: [Tensor Toolbox for MATLAB](https://www.tensortoolbox.org/index.html)

The Tensor Toolbox provides the classes and functions for manipulating dense, sparse, and structured tensors using MATLAB's object-oriented features. 

- **Tensor Types:** The Tensor Toolbox supports multiple tensor types, including dense, sparse, and symmetric tensors as well as specially structured tensors, such as Tucker format (core tensor plus factor matrices), Krusal format (stored as factor matrices), and sum format (sum of different types of tensors such as sparse plus rank-1 tensor in Kruskal format).
  
- **CP Decompositions:** CP methods such as alternating least squares, direct optimization, and weighted optimization (for missing data). Also alternative decompositions such as Poisson Tensor Factorization via alternating Poisson regression (APR), Generalized CP (GCP) tensor factorization, and symmetric CP tensor factorization. 

- **Tucker Decomposition:** Tucker methods including the higher-order SVD (HOSVD), the sequentially-truncated HOSVD (ST-HOSVD), and the higher-order orthogonal iteration (HOOI).

- **Eigenproblems:** Methods to solve the tensor eigenproblem including the shifted higher-order power method (SSHOPM) and the adaptive shift version (GEAP). 

- **Working with Tensors:** Creating test problems, tensor multiplication, collapsing and scaling tensors (useful in preprocessing), and more. 

- **Optimization Methods:** Standardized wrappers to make it simple to switch between several different optimization methods, including limited-memory BFGS quasi-Newton method and Adam (stochastic optimization).
  
## API Reference

For a comprehensive breakdown of all classes and methods, visit our [Documentation](https://pyttb.readthedocs.io/en/stable/index.html).

## Case Studies and Example Use Cases

Stay tuned for case studies and real-world examples of pyttb in action.

## Community and Support

Please visit our [Contact section](https://pyttb.readthedocs.io/en/stable/index.html) for support or questions. You can also refer to the [CONTRIBUTORS.md](https://github.com/sandialabs/pyttb/blob/main/CONTRIBUTORS.md) file to see the developers and contributors of pyttb.

## Versioning / Release Notes

For details on different versions and updates of pyttb, check our [Releases page](https://github.com/sandialabs/pyttb/releases).

## License Information

pyttb is released under the BSD 2-Clause License. You can view the details of the license [here](https://github.com/sandialabs/pyttb/blob/main/LICENSE).


---
