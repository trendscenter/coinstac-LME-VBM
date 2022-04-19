#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys
import npMatrix3d
import pandas as pd
import os
import itertools
from image_utils import nifti_to_data,gen_outputimages,gen_covBimage
import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm
from nilearn.image import concat_imgs
np.set_printoptions(threshold=sys.maxsize)

MASK = 'mask.nii'

'''
=============================================================================
The below function forms the matrices X, Y and Z.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- inputdir: base directory containing input csv files.
- fc: csv file containing fixed covariates, each row for an observation
- fs_vars: freesurfer regions to be used as response/dependent variables
- dep : csv file containing list of freesurfer stats file, each row for an
observation
- rf : csv file containing random factor levels for each observation (contains
one column as only 1 random factor)
- rc : csv file containing design matrix for the random factor
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - X: Fixed effects design matrix of dimension n times p
 - Y: The response matrix of dimension n times v
 - Z: The random effects design matrix of dimension n times q
 - ranfac: Vector containing site numbers of dimension n times 1
 - raneffs: Random effects design matrix of dimension n times 1
=============================================================================
'''
def form_XYZMatrices(inputdir,fc,dep,rf,rc,voxel_size):
    data_f = fc
    covariates=[]
    covariates.extend(['const'])
    covariates.extend(list(data_f.columns))

    data_f['isControl'] = data_f['isControl']*1
    cols_categorical = [col for col in data_f if data_f[col].dtype == object]
    cols_mono = [col for col in data_f if data_f[col].nunique() == 1]

    # Dropping columsn with unique values
    data_f = data_f.drop(columns=cols_mono)

    # Creating dummies on non-unique categorical variables
    cols_nodrop = set(cols_categorical) - set(cols_mono)
    data_f = pd.get_dummies(data_f, columns=cols_nodrop, drop_first=True)

    data_f = data_f.dropna(axis=0, how='any')
    data_f = data_f.to_numpy()

    X = data_f
    X = sm.add_constant(data_f)

    n=len(X)

    files = dep
    Y = nifti_to_data(inputdir,files,voxel_size)

    #raise Exception(Y)

    #ranfac = rf
    ranfac = rf.reshape((rf.shape[0], 1))
    nlevels = np.max(ranfac)
    #nlevels = int(nlevels)

    raneffs = rc

    Z = np.zeros([n, nlevels], dtype=int)

    for i in range(n):
        Z[i][ranfac[i]-1] = raneffs[i]

    return(X,Y,Z,ranfac,raneffs,covariates)

'''
=============================================================================
The below function generates the product matrices from matrices X, Y and Z.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
 - X: The design matrix of dimension n times p.
 - Y: The response vector of dimension v times n times 1*.
 - Z: The random effects design matrix of dimension n times q.
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - XtX: X transposed multiplied by X.
 - XtY: X transposed multiplied by Y.
 - XtZ: X transposed multiplied by Z.
 - YtX: Y transposed multiplied by X.
 - YtY: Y transposed multiplied by Y.
 - YtZ: Y transposed multiplied by Z.
 - ZtX: Z transposed multiplied by X.
 - ZtY: Z transposed multiplied by Y.
 - ZtZ: Z transposed multiplied by Z.
=============================================================================
'''
def prodMats3D(X,Y,Z):

    X=np.array(X)
    Y=np.transpose(np.array(Y))
    Y1=np.zeros([np.shape(Y)[0],np.shape(Y)[1],1])
    Y1[:,:,0]=Y
    Y=Y1
    Z=np.array(Z)

    # Work out the product matrices (non spatially varying)
    XtX = (X.transpose() @ X).reshape(1, X.shape[1], X.shape[1])
    XtY = X.transpose() @ Y
    XtZ = (X.transpose() @ Z).reshape(1, X.shape[1], Z.shape[1])
    YtX = XtY.transpose(0,2,1)
    YtY = Y.transpose(0,2,1) @ Y
    YtZ = Y.transpose(0,2,1) @ Z
    ZtX = XtZ.transpose(0,2,1)
    ZtY = YtZ.transpose(0,2,1)
    ZtZ = (Z.transpose() @ Z).reshape(1, Z.shape[1], Z.shape[1])

    # Return product matrices
    return(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ)

'''
=============================================================================
The below function extracts the parameters estimated in LME from paramVec and
reconstructs D .
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
 - paramVec: parameter estimates output from pSFS3D.
 - p: number of fixed effects.
 - v: number of freesurfer variables/regions
 - nlevels: number of levels of random factor
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - beta : The fixed effects parameter estimates for each fs region
 - sigma2 : The fixed effects variance estimate for each fs region
 - Ddict : unique element of random effects covariance matrix
 - D : The random effects covariance matrix estimate for each fs region
=============================================================================
'''
def get_parameterestimates(paramVec,p,v,nlevels,nraneffs):

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

     # Output beta estimate
    beta = paramVec[:, 0:p]

    # Output sigma2 estimate
    sigma2 = paramVec[:,p:(p+1),:]

    # Get the indices in the paramvector corresponding to D matrices
    IndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)//2) + p + 1)
    IndsDk = np.insert(IndsDk,0,p+1)

    # Reconstruct D
    Ddict = dict()
    # D as a dictionary

    Ddict[0] = npMatrix3d.vech2mat3D(paramVec[:,IndsDk[0]:IndsDk[1],:])

    # Full version of D
    D = npMatrix3d.getDfromDict3D(Ddict, nraneffs, nlevels)

    return(beta,sigma2,Ddict[0],D)

'''
=============================================================================
The below function generates the global Z matrix including random factors and
effects from all local sites.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- nlevels_persite : list containing number of levels of random factor for
    each site
- nlevels_global : total levels summed up for all local sites
- nlocalsites : number of local sites
- clientId : local site id
- ranfac : vector defining level for each observation
- raneffs : vector containing random effects
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
- Z : global random effects design matrix of dimension n times q.
=============================================================================
'''
def form_globalZMatrix(nlevels_persite,nlevels_global,nlocalsites,clientId,ranfac,raneffs):

    nlevels_persite = np.array(nlevels_persite)
    n = len(raneffs)
    Z = np.zeros([n, nlevels_global], dtype=int)
    col_start = 0

    for s in range(nlocalsites):
        if (clientId=='local'+str(s)):
            if s==0:
                col_start = 0
            else:
                for i in range(s):
                    col_start = col_start+nlevels_persite[i]


    ranfac = np.array(ranfac)
    for i in range(n):
        Z[i][col_start+ranfac[i][0]-1] = raneffs[i]

    return(Z)

'''
=============================================================================
The below functions save_results and save_parameters generates the output images, this includes,
- global stats and local stats
each of the stats contains
- parameter estimates and inference stats
- parameter estimates:
    beta
    sigma2
    vechD
- inference stats
    llh :
    resms
    covB
    tstats
    fstats
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- state_list
- dict_params
- covariates : vector containing random effects
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
- result_imgnames : list of resultant image names
=============================================================================
'''
def save_results(state_list,dict_params,covariates):
    result_imgnames = []
    params = ['sigmasquared','covRandomEffects','log-likelihood',
                'residualmeansquares','covBeta','tstats','fstats']
    for p in params:
        data = np.asarray(dict_params[p])
        if p!='tstats' and  p!='fstats':
            data = np.squeeze(data)
        temp = save_parameters(state_list,data,p,covariates)
        result_imgnames = result_imgnames+temp
    return(result_imgnames)


def save_parameters(state_list,data,p,covariates):
    outputdir = state_list["outputDirectory"]
    result_imgnames = []
    if p=='covBeta':
        dimCov = len(covariates)**2
        for d in range(dimCov):
            temp = np.asarray(data)
            image_fname = 'covBeta_'+str(d+1)
            temp = temp[:,d]
            gen_outputimages(state_list,temp,image_fname)
        output_file = gen_covBimage(outputdir,dimCov)
        result_imgnames.append(output_file)

    elif p=='tstats' or p=='fstats':
        if p=='tstats':
            stats = ['Beta','stderrorBeta','dof','t-stat','p-value']
        else:
            stats = ['dof','f-stat','p-value','rsquared']
        data = np.asarray(data)
        for t in range(data.shape[0]):
            for s in range(len(stats)):
                temp = np.asarray(data[t])
                image_fname = temp[0]+'_'+stats[s]
                temp = np.squeeze(np.asarray(temp[2][s]))
                gen_outputimages(state_list,temp,image_fname)
                result_imgnames.append(os.path.join(outputdir,image_fname+'.nii.gz'))

    else:
        image_fname = p
        temp = np.asarray(data)
        gen_outputimages(state_list,temp,image_fname)
        result_imgnames.append(os.path.join(outputdir,image_fname+'.nii.gz'))

    return(result_imgnames)

