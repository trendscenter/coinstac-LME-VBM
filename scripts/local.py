#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script includes the local computations for linear mixed effects model
with decentralized statistic calculation
"""

import json
import numpy as np
import sys
import regression as reg
import lme_utils
import data_utils
import os
from image_utils import average_nifti
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

"""
============================================================================
The below function calculates average nifti volume for each local site
----------------------------------------------------------------------------
This function takes in the following inputs in args:
----------------------------------------------------------------------------
- input
    fixed_covariates : csv file containing fixed covariates, each row for an 
    observation
    dependents : csv file containing list of freesurfer stats file, each row 
    for an observation
    random_factor : csv file containing random factor levels for each 
    observation (contains one column as only 1 random factor)
    random_covariates : csv file containing design matrix for the random
    factor
    contrasts : list of contrasts. Contrast vectors to be tested. 
    Each contrast should contain the fields:
    name: A name for the contrast. i.e. Contrast1.
    vector: A vector for the contrast. 
    It must be one dimensional for a T test and two dimensional for an F test 
    For eg., [1, 0, 0] (T contrast) or [[1, 0, 0],[0,1,0]] (F contrast)
    threshold : threshold value for generating mask
    voxel_size : voxel size of the images
- state
    baseDirectory : '/input/local<site>/simulatorRun'
    transferDirectory : '/transfer/local<site>/simulatorRun'
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    threshold : threshold value for generating mask
    voxel_size : voxel size of the images
    avg_nifti : average nifti volume
    computation_phase : local_0
- cache:
    fixed_covariates : csv file containing fixed covariates, each row for 
    an observation
    dependents : csv file containing list of freesurfer stats file, each 
    row for an observation
    random_factor : csv file containing random factor levels for each 
    observation (contains one column as only 1 random factor)
    random_covariates : csv file containing design matrix for the random
    factor
    contrasts : list of contrasts. Contrast vectors to be tested.
============================================================================
"""
def local_0(args):
    
    input_list = args["input"]
    state_list = args["state"]

    dep = input_list['dependents']
    threshold = input_list["mask_threshold"]
    voxel_size = input_list["voxel_size"]
    outputdir = state_list["transferDirectory"]
    inputdir = state_list['baseDirectory']
    
    average_nifti(inputdir,dep,outputdir)   

    output_dict = {
        "threshold": threshold,
        "voxel_size": voxel_size,
        "avg_nifti": "avg_nifti.nii",
        "computation_phase": "local_0"
    }
    cache_dict = {
        'fixed_covariates': input_list['fixed_covariates'],
        'dependents': dep,
        'random_factor': input_list['random_factor'],
        'random_covariates': input_list['random_covariates'],
        'contrasts': input_list['contrasts'],
        "voxel_size": voxel_size,
    }

    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output_dict)

"""
============================================================================
The below function does the following tasks
1. read the dependent variables, fixed and random covariates from csv files
and forms the X,Y and Z matrices
2. calculate the product matrices
3. solves LME model using pseudo Simplified Fisher Scoring algorithm and 
outputs the parameter estimates and inference results
4. generates resultant images for each of the parameters
----------------------------------------------------------------------------
This function takes in the following inputs in args:
----------------------------------------------------------------------------
- input
    mask : average of masks from all local sites
    computation_phase : remote_0
- cache
    fixed_covariates : csv file containing fixed covariates, each row for an 
    observation
    dependents : csv file containing list of freesurfer stats file, each row 
    for an observation
    random_factor : csv file containing random factor levels for each 
    observation (contains one column as only 1 random factor)
    random_covariates : csv file containing design matrix for the random
    factor
    contrasts : list of contrasts. Contrast vectors to be tested. 
    Each contrast should contain the fields:
    name: A name for the contrast. i.e. Contrast1.
    vector: A vector for the contrast. 
    It must be one dimensional for a T test and two dimensional for an F test 
    For eg., [1, 0, 0] (T contrast) or [[1, 0, 0],[0,1,0]] (F contrast)
    voxel_size : voxel size of the images
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    nlevels : list containing number of levels for each random factor, as we 
    have only one random factor, nlevels is a single value
    nobservns : number of observations
    local_result_images : list of result images from local sites
    computation_phase : local_1
- cache:
    X : fixed effects design matrix
    Y : response matrix
    ranfac : vector defining level for each observation
    raneffs : vector containing random effects
    covariates : list of fixed covariates labels
============================================================================
"""
def local_1(args):
    
    cache_list = args['cache']
    state_list = args['state']

    cache_dir = state_list["cacheDirectory"]
    inputdir = state_list['baseDirectory']
    fc = cache_list['fixed_covariates']
    dep = cache_list['dependents']
    rf = cache_list['random_factor']
    rc = cache_list['random_covariates']
    contrasts=cache_list['contrasts']
    voxel_size = cache_list["voxel_size"]

    [X,Y,Z,ranfac,raneffs,covariates] = lme_utils.form_XYZMatrices(inputdir,fc,dep,rf,rc,voxel_size)

    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = lme_utils.prodMats3D(X,Y,Z)

    n = len(X)
    nlevels = np.array([np.shape(Z)[1]])
    nraneffs = np.array([1])
    tol = 1e-6
    paramVec = reg.pSFS3D(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nlevels, nraneffs,tol,n)
    
    nfixeffs = np.shape(X)[1]
    ndepvars = np.shape(YtX)[0]
    [beta,sigma2,vechD,D] = lme_utils.get_parameterestimates(paramVec,nfixeffs,ndepvars,nlevels,nraneffs)
    
    
    prod_matrices = [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
    [llh,resms,covB,tstats,fstats] = reg.cal_inference(prod_matrices,n,nfixeffs,ndepvars,nlevels,
                                                        nraneffs,beta,sigma2,D,contrasts)

    local_dict = {'sigmasquared':sigma2, 'covRandomEffects':vechD,
                    'log-likelihood':llh, 'residualmeansquares':resms,
                    'covBeta':covB,'tstats':tstats,'fstats':fstats}
    
    result_imgnames = lme_utils.save_results(state_list,local_dict,covariates)

    # Writing covariates and dependents to cache as files
    data_utils.saveBin(os.path.join(cache_dir, 'X.npy'), X)
    data_utils.saveBin(os.path.join(cache_dir, 'Y.npy'), Y)

    computation_output_dict = {
        'output': 
        {
            'nlevels': nlevels.tolist(),
            'nobservns': n,
            'local_result_images': result_imgnames,
            'computation_phase': 'local_1'
        },
        'cache': 
        {
            'X': 'X.npy',
            'Y': 'Y.npy',
            'ranfac': ranfac.tolist(),
            'raneffs': raneffs,
            'covariates': covariates,
        }
    }

    return json.dumps(computation_output_dict)


"""
============================================================================
The below function does the following tasks
1. form Z matrix
2. calculate product matrices
3. send the product matrices to remote_2
----------------------------------------------------------------------------
This function takes in the following inputs in args:
----------------------------------------------------------------------------
- input :
    nlevels_persite : list containing number of levels of random factor for 
    each site
    nlevels_global : total levels summed up for all local sites
    nlocalsites : number of local sites
    computation_phase : remote_1
- cache:
    X : fixed effects design matrix
    Y : response variables
    ranfac : vector defining level for each observation
    raneffs : vector containing random effects
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    XtransposeX_local
    XtransposeY_local
    XtransposeZ_local
    YtransposeX_local
    YtransposeY_local
    YtransposeZ_local
    ZtransposeX_local
    ZtransposeY_local
    ZtransposeZ_local
    contrasts : list of contrasts
    computation_phase : local_2
============================================================================
"""
def local_2(args):
    
    cache_list = args['cache']
    input_list = args['input']
    state_list = args['state']

    ranfac = cache_list['ranfac']
    raneffs = cache_list['raneffs']
    covariates = cache_list['covariates']
    cache_dir = state_list["cacheDirectory"]
    transfer_dir = state_list["transferDirectory"]

    X = data_utils.loadBin(os.path.join(cache_dir, cache_list['X']))
    Y = data_utils.loadBin(os.path.join(cache_dir, cache_list['Y']))

    n = len(X)
    nlevels_persite = input_list['nlevels_persite']
    nlevels_global = input_list['nlevels_global']
    nlocalsites = input_list['nlocalsites']
    clientId = args['state']['clientId']

    Z = lme_utils.form_globalZMatrix(nlevels_persite,nlevels_global,nlocalsites,clientId,ranfac,raneffs)

    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = lme_utils.prodMats3D(X,Y,Z)

    prod_matrices = [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
    prod_matrices_name = ['XtX','XtY','XtZ','YtX','YtY','YtZ','ZtX','ZtY','ZtZ']

    for p in range(len(prod_matrices)):
        data_utils.saveBin(os.path.join(transfer_dir, prod_matrices_name[p]+'.npy'),prod_matrices[p])

    computation_output_dict = {
        'output': {
            'XtransposeX_local': 'XtX.npy',
            'XtransposeY_local': 'XtY.npy',
            'XtransposeZ_local': 'XtZ.npy',
            'YtransposeX_local': 'YtX.npy',
            'YtransposeY_local': 'YtY.npy',
            'YtransposeZ_local': 'YtZ.npy',
            'ZtransposeX_local': 'ZtX.npy',
            'ZtransposeY_local': 'ZtY.npy',
            'ZtransposeZ_local': 'ZtZ.npy',
            'contrasts': cache_list['contrasts'],
            'covariates': covariates,
            'computation_phase': 'local_2',
        }
    }
    return json.dumps(computation_output_dict)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())

    phase_key = list(data_utils.list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_0(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_0' in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_1' in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Error occurred at Local')
