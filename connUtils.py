#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:14:01 2019

@author: Or Duek
This file should contain all connectivity analysis functions so we could load from it to other files
"""
import numpy as np
import scipy

def removeVars (confoundFile):
    # this method takes the csv regressors file (from fmriPrep) and chooses a few to confound. You can change those few
    import pandas as pd
    confound = pd.read_csv(confoundFile,sep="\t", na_values="n/a")
    finalConf = confound[['csf', 'white_matter', 'framewise_displacement',
                          'a_comp_cor_00', 'a_comp_cor_01',	'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 
                        'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']] # can add 'global_signal' also
     # change NaN of FD to zero
    finalConf = np.array(finalConf)
    finalConf[0,2] = 0 # if removing FD than should remove this one also
    return finalConf

# build method for creating time series for subjects 
def timeSeries(func_files, confound_files, atlas_filename):
    # This function receives a list of funcional files and a list of matching confound files
    # and outputs an array
    from nilearn.input_data import NiftiLabelsMasker
    # define masker here
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, smoothing_fwhm = 6,
                        memory="/media/Data/nilearn",t_r=1, verbose=5, high_pass=.01 , low_pass = .1) # As it is task based we dont' bandpassing high_pass=.01 , low_pass = .1)
    total_subjects = [] # creating an empty array that will hold all subjects matrix 
    # This function needs a masker object that will be defined outside the function
    for func_file, confound_file in zip(func_files, confound_files):
        print(f"proccessing file {func_file}") # print file name
        confoundClean = removeVars(confound_file)
        confoundArray = confoundClean#confoundClean.values
        time_series = masker.fit_transform(func_file, confounds=confoundArray)
        #time_series = extractor.fit_transform(func_file, confounds=confoundArray)
        #masker.fit_transform(func_file, confoundArray)
        total_subjects.append(time_series)
    return total_subjects

def timeSeriesSingle(func_file, confound_file, atlas_filename):
    # this function receives one functional and one confound file and returns one time-series
    from connUtils import removeVars
    from nilearn.input_data import NiftiLabelsMasker
    # define masker here
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, smoothing_fwhm = 6,
                        memory="/media/Data/nilearn",t_r=1, verbose=5, high_pass=.01 , low_pass = .1) # As it is task based we dont' bandpassing high_pass=.01 , low_pass = .1)
    # This function needs a masker object that will be defined outside the function
    confoundClean = removeVars(confound_file)
    confoundArray = confoundClean#confoundClean.values
    time_series = masker.fit_transform(func_file, confounds=confoundArray)
    return time_series

# contrasting two timePoints
def contFuncs(time_series1, time_series2):
    twoMinusOneMat = []
    for scanMatrix, scanMatrix2 in zip(time_series1, time_series2):
        a = scanMatrix2 - scanMatrix
        twoMinusOneMat.append(a)
    return np.array(twoMinusOneMat)

# create correlation matrix per subject
def createCorMat(time_series):
    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind='correlation') # can choose partial - it might be better        
    # create correlation matrix for each subject
    fullMatrix = []
    for time_s in time_series:
        correlation_matrix = correlation_measure.fit_transform([time_s])[0]
        fullMatrix.append(correlation_matrix)
    return fullMatrix


## Seed based functions
def createDelta(func_files1, func_files2, mask_img):
    from nilearn.input_data import NiftiMasker
    
    # here I use a masked image so all will have same size
    nifti_masker = NiftiMasker(
        mask_img= mask_img,
        smoothing_fwhm=6,
        memory='nilearn_cache', memory_level=1, verbose=2)  # cache options
    fmri_masked_ses1 = nifti_masker.fit_transform(func_files1)
    fmri_masked_ses2 = nifti_masker.fit_transform(func_files2)
    ###
    from nilearn import input_data
    brainMasker = input_data.NiftiMasker(
            smoothing_fwhm=6,
            detrend=True, standardize=True,
            t_r=1.,
            memory='/media/Data/nilearn', memory_level=1, verbose=2)
    brainMasker.fit(func_files1)

    ####
    deltaCor_a = fmri_masked_ses2 - fmri_masked_ses1
    print (f'Shape is: {deltaCor_a.shape}')

    # run paired t-test 
    testDelta = scipy.stats.ttest_rel(fmri_masked_ses1, fmri_masked_ses2) 
    print (f'Sum of p values < 0.005 is {np.sum(testDelta[1]<0.005)}')
    
    
    return deltaCor_a, testDelta # return the delta correlation and the t-test array

def createZimg(deltaCor, scriptName, seedName):
    from nilearn import input_data
    brainMasker = input_data.NiftiMasker(
            smoothing_fwhm=6,
            detrend=True, standardize=True,
            t_r=1.,
            memory='/media/Data/nilearn', memory_level=1, verbose=2)
    # mean across subjects
    mean_zcor_Delta = np.mean(deltaCor,0)
    mean_zcor_img_delta = brainMasker.inverse_transform(mean_zcor_Delta.T)
    # save it as file
    mean_zcor_img_delta.to_filename(
        '/home/or/kpe_conn/%s_seed_%s_delta_z.nii.gz' %(scriptName,seedName))
    
    return mean_zcor_img_delta, mean_zcor_Delta # returns the image and the array 

## now create a function to do FDR thresholding
def fdrThr(testDelta, mean_zcor_Delta, alpha, brain_masker):
    from statsmodels.stats import multitest
    # we need to reshape the test p-values array to create 1D array
    #b = np.reshape(np.array(testDelta[1]), -1)
    fdr_mat = multitest.multipletests(testDelta[1], alpha=alpha, method='fdr_bh', is_sorted=False, returnsorted=False)
    #fdr_mat = multitest.fdrcorrection(testDelta[1], alpha=0.7, method='indep', is_sorted=False)
    np.sum(fdr_mat[1]<0.05)
    corr_mat_thrFDR = np.array(mean_zcor_Delta)
    corr_mat_thrFDR = np.reshape(corr_mat_thrFDR, -1)
    corr_mat_thrFDR[fdr_mat[0]==False] = 0
   
    # now I can treshold the mean matrix
    numNonZeroDelta = np.count_nonzero(corr_mat_thrFDR)
    print (f'Number of voxels crossed the FDR thr is {numNonZeroDelta}')
    # transofrm it back to nifti
    nifti_fdr_thr = brain_masker.inverse_transform(corr_mat_thrFDR.T)
    return corr_mat_thrFDR, nifti_fdr_thr # return matrix after FDR and nifti file
                           
