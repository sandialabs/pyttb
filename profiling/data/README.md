# Create data for profiling

## Scripts

- `create_profile_data.m`: MATLAB script for creating data

## Generating Data

In MATLAB (with the [Tensor Toolbox](https://www.tensortoolbox.org/) installed):

```matlab
>> create_profile_data
```

The following files are generated:
```
sptensor_sparse_continuous_size_1000_3_3_rng_1.tns
sptensor_sparse_continuous_size_1000_3_3_rng_2.tns
sptensor_sparse_continuous_size_1000_3_3_rng_3.tns
sptensor_sparse_continuous_size_10_8_6_4_rng_1.tns
sptensor_sparse_continuous_size_10_8_6_4_rng_2.tns
sptensor_sparse_continuous_size_10_8_6_4_rng_3.tns
sptensor_sparse_continuous_size_20_16_12_rng_1.tns
sptensor_sparse_continuous_size_20_16_12_rng_2.tns
sptensor_sparse_continuous_size_20_16_12_rng_3.tns
sptensor_sparse_integer_size_1000_3_3_rng_1.tns
sptensor_sparse_integer_size_1000_3_3_rng_2.tns
sptensor_sparse_integer_size_1000_3_3_rng_3.tns
sptensor_sparse_integer_size_10_8_6_4_rng_1.tns
sptensor_sparse_integer_size_10_8_6_4_rng_2.tns
sptensor_sparse_integer_size_10_8_6_4_rng_3.tns
sptensor_sparse_integer_size_20_16_12_rng_1.tns
sptensor_sparse_integer_size_20_16_12_rng_2.tns
sptensor_sparse_integer_size_20_16_12_rng_3.tns
tensor_dense_continuous_size_1000_3_3_rng_1.tns
tensor_dense_continuous_size_1000_3_3_rng_2.tns
tensor_dense_continuous_size_1000_3_3_rng_3.tns
tensor_dense_continuous_size_10_8_6_4_rng_1.tns
tensor_dense_continuous_size_10_8_6_4_rng_2.tns
tensor_dense_continuous_size_10_8_6_4_rng_3.tns
tensor_dense_continuous_size_20_16_12_rng_1.tns
tensor_dense_continuous_size_20_16_12_rng_2.tns
tensor_dense_continuous_size_20_16_12_rng_3.tns
tensor_dense_integer_size_1000_3_3_rng_1.tns
tensor_dense_integer_size_1000_3_3_rng_2.tns
tensor_dense_integer_size_1000_3_3_rng_3.tns
tensor_dense_integer_size_10_8_6_4_rng_1.tns
tensor_dense_integer_size_10_8_6_4_rng_2.tns
tensor_dense_integer_size_10_8_6_4_rng_3.tns
tensor_dense_integer_size_20_16_12_rng_1.tns
tensor_dense_integer_size_20_16_12_rng_2.tns
tensor_dense_integer_size_20_16_12_rng_3.tns
```

## Loading Data

In MATLAB, load the data as follows:

```matlab
>> X = import_data('sptensor_sparse_continuous_size_1000_3_3_rng_1.tns');
```

### Running Decomposition Algorithms

The data generated can be used with the following decomposition algorithms:

- `sptensor_sparse_continuous*`: cp_als, hosvd, tucker_als
- `sptensor_sparse_integer*`: cp_apr, cp_als, hosvd, tucker_als
- `tensor_dense_continuous*`: cp_als, hosvd, tucker_als
- `tensor_dense_integer*`: cp_apr, cp_als, hosvd, tucker_als
