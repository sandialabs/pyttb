# Profiling PYTTB
Profiling support for pyttb is limited at this time with a focus on feature coverage.
Proposals and contributions welcome!
However, this directory contains minimal initial support.

## Setup
* `pip install .[dev,profiling]`
    * Get the dev and profiling dependencies for a local install from source
* Generate the profiling data in `./data`
    * Unfortunately this currently depends on MATLAB and the MATLAB TensorToolbox
* Run the `algorithms_profiling.ipynb` which currently takes ~10s of minutes
