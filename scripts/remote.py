#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script includes the remote computations for linear mixed effects model
with decentralized statistic calculation
"""

import json
import os
import sys
import numpy as np
import regression as reg
import lme_utils
import data_utils
from image_utils import calculate_mask

"""
============================================================================
The below function calculates the global mask
----------------------------------------------------------------------------
This function takes in the following inputs in args['input']:
----------------------------------------------------------------------------
- threshold : threshold value for generating mask
- voxel_size : voxel size of the images
- avg_nifti : average nifti volume
- computation_phase : local_0
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    mask : average of masks from all local sites
    computation_phase : remote_0
============================================================================
"""
def remote_0(args):

    calculate_mask(args)

    computation_output_dict = {
        "output": {
            "mask": 'mask.nii',
            "computation_phase": "remote_0"
        }
    }
    return json.dumps(computation_output_dict)

"""
============================================================================
The below function does the following tasks
1. read the number of levels of random factor and number of observations, for
each local site
2. calculate the total number of levels and total observations
3. send the values to local_1
----------------------------------------------------------------------------
This function takes in the following inputs in args['input']:
----------------------------------------------------------------------------
- nlevels : list containing number of levels for each random factor, as we 
    have only one random factor, nlevels is a single value
- nobservns : number of observations per local site
- local_result_images : list of result images from local sites
- computation_phase : local_1
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    nlevels_persite : list containing number of levels of random factor for 
    each site
    nlevels_global : total levels summed up for all local sites
    nlocalsites : number of local sites
    computation_phase : remote_1
- cache:
    nlevels_global : total levels summed up for all local sites
    nobservns_global : total observations summed up for all local sites
    local_result_images : list of result images from local sites
============================================================================
"""
def remote_1(args):
    input_list = args['input']
    
    nlevels_persite = [input_list[site]['nlevels'] for site in input_list]
    nlevels_global = np.sum(np.array(nlevels_persite))
    nobservns = [input_list[site]['nobservns'] for site in input_list]
    nobservns_global = np.sum(np.array(nobservns))
    
    result_imgnames_local = [input_list[site]['local_result_images'] for site in input_list]

    computation_output_dict = {
        'cache': 
        {
            'nlevels_global': nlevels_global.tolist(),
            'nobservns_global': nobservns_global.tolist(),
            'local_result_images' : result_imgnames_local
        },
        'output': 
        {
            'nlevels_persite': nlevels_persite,
            'nlevels_global': nlevels_global.tolist(),
            'nlocalsites': len(input_list),
            'computation_phase': 'remote_1'
        }
    }

    return json.dumps(computation_output_dict)


"""
============================================================================
The below function does the following tasks
1. solves LME model using pseudo Simplified Fisher Scoring algorithm for 
aggregate data to find global parameter estimates
2. calculate global inference parameters
----------------------------------------------------------------------------
This function takes in the following inputs in args:
----------------------------------------------------------------------------
- input :
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
- cache:
    nlevels_global : total levels summed up for all local sites
    nobservns_global : total observations summed up for all local sites
    local_result_images : list of result images from local sites
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    computation_output_dict : dict containing names of all resultant images
============================================================================
"""
def remote_2(args):
    state_list = args['state']
    input_list = args['input']
    cache_list = args['cache']
    input_dir = state_list["baseDirectory"]
    transfer_dir = state_list["transferDirectory"]
    
    prod_matrices = ['XtransposeX_local','XtransposeY_local','XtransposeZ_local',
                    'YtransposeX_local','YtransposeY_local','YtransposeZ_local',
                    'ZtransposeX_local','ZtransposeY_local','ZtransposeZ_local']
    prod_matrices_vars=[]
    for p in prod_matrices:
        temp=0
        for site in input_list:
            temp=temp+np.array(data_utils.loadBin(os.path.join(input_dir,
                                site,input_list[site][p])))

        prod_matrices_vars.append(temp)

    XtX = prod_matrices_vars[0]
    XtY = prod_matrices_vars[1]
    XtZ = prod_matrices_vars[2]
    YtX = prod_matrices_vars[3]
    YtY = prod_matrices_vars[4]
    YtZ = prod_matrices_vars[5]
    ZtX = prod_matrices_vars[6]
    ZtY = prod_matrices_vars[7]
    ZtZ = prod_matrices_vars[8]
    
    nlevels_global = np.array([cache_list['nlevels_global']])
    nobservns_global = cache_list['nobservns_global']
    tol = 1e-6
    nraneffs = np.array([1])
    
    # Run Pseudo Simplified Fisher Scoring
    paramVec = reg.pSFS3D(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ,  
                            nlevels_global,nraneffs,tol,nobservns_global)
    
    nfixeffs = np.shape(XtX)[1]
    ndepvars = np.shape(XtY)[0]
    [beta,sigma2,vechD,D] = lme_utils.get_parameterestimates(paramVec,nfixeffs,ndepvars,
                                                                nlevels_global,nraneffs)
                                                                
    prod_matrices = [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
    contrasts = [input_list[site]['contrasts'] for site in input_list]
    con = contrasts[0]
    [llh,resms,covB,tstats,fstats] = reg.cal_inference(prod_matrices,nobservns_global,
                                                        nfixeffs,ndepvars,nlevels_global,
                                                        nraneffs,beta,sigma2,D,con)
    covariates = input_list['local0']['covariates']
    global_dict = {'sigmasquared':sigma2, 'covRandomEffects':vechD,
                    'log-likelihood':llh, 'residualmeansquares':resms,
                    'covBeta':covB, 'tstats':tstats,'fstats':fstats}
    
    result_imgnames_global = lme_utils.save_results(state_list,global_dict,covariates)
    
    output_dict = {"global_result_images": result_imgnames_global,
                    "local_result_images":cache_list['local_result_images']}

    computation_output_dict = {"output": output_dict, "success": True}
    return json.dumps(computation_output_dict)

if __name__ == '__main__':

    PARAM_DICT = json.loads(sys.stdin.read())
    PHASE_KEY = list(data_utils.list_recursive(PARAM_DICT, 'computation_phase'))

    if "local_0" in PHASE_KEY:
        sys.stdout.write(remote_0(PARAM_DICT))
    elif "local_1" in PHASE_KEY:
        sys.stdout.write(remote_1(PARAM_DICT))
    elif "local_2" in PHASE_KEY:
        sys.stdout.write(remote_2(PARAM_DICT))
    else:
        raise ValueError("Error occurred at Remote")
