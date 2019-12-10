#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:21:47 2019

@author: Or Duek
Check Aging data connectivity
"""

import os
import pandas as pd
from nilearn import plotting
from connUtils import removeVars, timeSeriesSingle, createCorMat


## set atlas (Here I use Yeo, but can use Shen or any other)

atlas_filename = '/home/or/Downloads/1000subjects_reference_Yeo/Yeo_JNeurophysiol11_SplitLabels/MNI152/Yeo2011_17Networks_N1000.split_components.FSL_MNI152_1mm.nii.gz'
atlas_labes = pd.read_csv('/home/or/Downloads/1000subjects_reference_Yeo/Yeo_JNeurophysiol11_SplitLabels/Yeo2011_17networks_N1000.split_components.glossary.csv')
coords = coords = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)
# take one subjects file 
# for this one I use the AROMA non aggresive one, as it is already "cleanded" (i.e. no need to remove confounds)
func_file = '/media/Data/Aging/Preprocessed_data/aging_output/fmriprep/sub-010/ses-1/func/sub-010_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confound_file = '/media/Data/Aging/Preprocessed_data/aging_output/fmriprep/sub-010/ses-1/func/sub-010_ses-1_task-rest_desc-confounds_regressors.tsv'


timeSer= timeSeriesSingle(func_file, confound_file, atlas_filename)


from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation') # can choose partial - it might be better    
correlation_matrix = correlation_measure.fit_transform([timeSer])[0]


plotting.plot_matrix(correlation_matrix)

plotting.plot_connectome(correlation_matrix, coords,
                         title='Correlation Matrix', edge_threshold="95%")