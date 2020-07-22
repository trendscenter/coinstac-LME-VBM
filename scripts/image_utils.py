#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script includes the NIfTI computations
"""

import os

import numpy as np
import pandas as pd
from nilearn import plotting
# from nilearn import image
import nibabel as nib
from nilearn.image import resample_img, resample_to_img

MASK = 'mask.nii'
MNI_TEMPLATE = '/computation/templates/MNI152_T1_1mm_brain.nii'

'''
=============================================================================
The below function calculates the average nifti volume for each local site
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- inputdir : basedirectory containing all nifti images
- dep : csv file containing list of VBM files
- outputdir : directory to save output average image
=============================================================================
'''
def average_nifti(inputdir,dep,outputdir):
    files = pd.read_csv(os.path.join(inputdir,dep))
    files = files['VBM_files']
    Y=[]
    appended_data = 0
    for image in files:
        image_data = nib.load(os.path.join(inputdir, image)).dataobj[:]
        if np.all(np.isnan(image_data)) or np.count_nonzero(
                image_data) == 0 or image_data.size == 0:
            files = files.drop(image)
        else:
            appended_data += image_data
    
    sample_image = nib.load(os.path.join(inputdir, image))
    header = sample_image.header
    affine = sample_image.affine

    avg_nifti = appended_data / len(files)

    clipped_img = nib.Nifti1Image(avg_nifti, affine, header)
    output_file = os.path.join(outputdir, 'avg_nifti.nii')
    nib.save(clipped_img, output_file)

'''
=============================================================================
The below function calculates the global mask by averaging masks across
local sites and saves mask image in cache and transfer directory of remote
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- args : input and state variables
=============================================================================
'''
def calculate_mask(args):
    """Calculates the average of all masks
    """
    input_ = args["input"]
    state_ = args["state"]
    input_dir = state_["baseDirectory"]
    cache_dir = state_["cacheDirectory"]
    output_dir = state_["transferDirectory"]
    
    site_ids = input_.keys()
    avg_of_all = sum([
        nib.load(os.path.join(input_dir, site,
                              input_[site]['avg_nifti'])).get_fdata()
        for site in input_
    ]) / len(site_ids)

    # Threshold binarizer
    user_id = list(input_)[0]
    threshold = input_[user_id]["threshold"]
    voxel_size = input_[user_id]["voxel_size"]

    mask_info = avg_of_all > threshold

    principal_image = nib.load(
        os.path.join(input_dir, user_id, input_[user_id]['avg_nifti']))
    header = principal_image.header
    affine = principal_image.affine

    clipped_img = nib.Nifti1Image(mask_info, affine, header)
    mni_image = os.path.join('/computation/templates', MNI_TEMPLATE)

    reoriented_mni = resample_to_img(mni_image,
                                     clipped_img,
                                     interpolation='linear')
    downsampled_mni = resample_img(reoriented_mni,
                                   target_affine=np.eye(3) * voxel_size,
                                   interpolation='linear')

    downsampled_mask = resample_to_img(clipped_img,
                                       downsampled_mni,
                                       interpolation='nearest')

    output_file1 = os.path.join(output_dir, 'mask.nii')
    output_file2 = os.path.join(cache_dir, 'mask.nii')
    output_file3 = os.path.join(output_dir, 'mni_downsampled.nii')
    output_file4 = os.path.join(cache_dir, 'mni_downsampled.nii')

    nib.save(downsampled_mask, output_file1)
    nib.save(downsampled_mask, output_file2)
    nib.save(downsampled_mni, output_file3)
    nib.save(downsampled_mni, output_file4)

'''
=============================================================================
The below function forms the Y matrix by extracting the voxels corresponding
to mask from the nifti images
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- inputdir : basedirectory containing all nifti images
- files : VBM files
- voxel_size : voxel size of the images
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
- y : y matrix
=============================================================================
'''
def nifti_to_data(inputdir,files,voxel_size):
    try:
        mask_data = nib.load(os.path.join(inputdir,
                                          MASK)).get_fdata()
        mask_dim = mask_data.shape
    except FileNotFoundError:
        raise Exception("Missing Mask at " + args["state"]["clientId"])

    mni_image = os.path.join(inputdir,
                             'mni_downsampled.nii')

    y = np.zeros((len(files), np.count_nonzero(mask_data)), dtype='f8')
    for index, image in enumerate(files):
        input_file = os.path.join(inputdir, image)
        if nib.load(input_file).header.get_zooms()[0] == voxel_size:
            image_data = nib.load(input_file).get_data()
        else:
            clipped_img = resample_to_img(input_file, mni_image)
            image_data = clipped_img.get_data()

        a = []
        for slicer in range(mask_dim[-1]):
            img_slice = image_data[slicer, ...]
            msk_slice = mask_data[slicer, ...]
            a.extend(img_slice[msk_slice > 0].tolist())

        y[index, :] = a
    return y

'''
=============================================================================
The below function generates resultant nifti images for each of the parameter 
estimates and inference statistics and generates a png image for a 
representative slice of the result images
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- state_list : basedirectory containing all nifti images
- data : parameter estimates
- image_fname : name of result image
=============================================================================
'''
def gen_outputimages(state_list,data,image_fname):
    images_folder = state_list["outputDirectory"]
    try:
        mask = nib.load(os.path.join(state_list["baseDirectory"], MASK))
    except FileNotFoundError:
        mask = nib.load(os.path.join(state_list["cacheDirectory"], MASK))

    new_data = np.zeros(mask.shape)
    new_data[mask.get_data() > 0] = data

    clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
    output_file = os.path.join(images_folder, image_fname)
    nib.save(clipped_img, output_file+'.nii.gz')

    plotting.plot_stat_map(clipped_img,
                            output_file=output_file,
                            display_mode='ortho',
                            colorbar=True)

'''
=============================================================================
The below function generates a single 4D covariance beta image by 
concatenating individual 3d covariance beta images
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- outputdir : directory containing 3d covariance beta images
- dimCov : number of 3d covariance beta images
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
- output_file : name of the concatenated 4D image ('covar_beta.nii.gz')
=============================================================================
'''
def gen_covBimage(outputdir,dimCov):
    output_files = [os.path.join(outputdir,'covBeta_'+str(d+1)+'.nii.gz') 
                        for d in range(dimCov)]
    output_file = os.path.join(outputdir, 'covBeta.nii.gz')
    ni2_funcs = [nib.Nifti2Image.from_image(nib.load(func)) for func in output_files]
    ni2_concat = nib.concat_images(ni2_funcs)
    ni2_concat.to_filename(output_file)
    for file in output_files:
        os.remove(file)
    return(output_file)