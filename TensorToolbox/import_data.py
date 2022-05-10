# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np
import os
import TensorToolbox as ttb

def import_data(filename):

    # Check if file exists
    if not os.path.isfile(filename):
        assert False, f"File path {filename} does not exist."

    # import
    fp = open(filename, 'r')

    # tensor type should be on the first line
    # valid: tensor, sptensor, matrix, ktensor
    data_type = import_type(fp)

    if data_type not in ['tensor','sptensor','matrix','ktensor']:
        assert False, f"Invalid data type found: {data_type}"
    
    if data_type == 'tensor':
    
        assert False, f"{data_type} is not currently allowed"
 
    elif data_type == 'sptensor':
    
        shape = import_shape(fp)
        nz = import_nnz(fp)
        subs, vals = import_sparse_array(fp, len(shape), nz)
        return ttb.sptensor().from_data(subs, vals, shape)
 
    elif data_type == 'matrix':         

        assert False, f"{data_type} is not currently allowed"
        
    elif data_type == 'ktensor':

        shape = import_shape(fp)
        #print(f"shape: {shape}")
        r = import_rank(fp)
        #print(f"rank: {r}")
        weights = np.array(fp.readline().strip().split(' '),dtype="float")
        #print(f"weights: {weights}")
        factor_matrices = []
        for n in range(len(shape)):
             fac_type = fp.readline().strip()
             #print(f"fac_type: {fac_type}")
             fac_shape = import_shape(fp)
             #print(f"fac_shape: {fac_shape}")
             fac = np.zeros(fac_shape, dtype="float")
             for r in range(fac_shape[0]):
                 fac[r,:] = fp.readline().strip().split(' ')
                 #print(f"fac: {fac}")
             factor_matrices.append(fac)
        return ttb.ktensor().from_data(weights, factor_matrices)
    
    # Close file
    fp.close()

def import_type(fp):
    # Import IO data type
    return fp.readline().strip().split(' ')[0]

def import_shape(fp):
    # Import the shape of something from a file
    n = int(fp.readline().strip().split(' ')[0])
    shape = [int(d) for d in fp.readline().strip().split(' ')]
    if len(shape) != n:
        assert False, "Imported dimensions are not of expected size"
    return tuple(shape)

def import_nnz(fp):
    # Import the size of something from a file
    return int(fp.readline().strip().split(' ')[0])

def import_rank(fp):
    # Import the rank of something from a file
    return int(fp.readline().strip().split(' ')[0])

def import_sparse_array(fp, n, nz):
    # Import sparse data subs and vals from coordinate format data
    subs = np.zeros((nz, n), dtype='int64')
    vals = np.zeros((nz, 1))
    for k in range(nz):
        line = fp.readline().strip().split(' ')
        subs[k,:] = line[:-1]
        vals[k,0] = line[-1]
    return subs, vals
