import os

PROJECT_PATH     = os.path.dirname(os.path.abspath(__name__))
DATA_PATH        = os.path.join(os.getcwd(), "data")
T1_PATH          = os.path.join(DATA_PATH, "T1w.nii")
MASK_PATH        = os.path.join(DATA_PATH, "mask.nii")
FMRI_PATH        = os.path.join(DATA_PATH, "fMRI.nii")
MNI_PATH         = os.path.join(DATA_PATH, "MNI152_T1_2mm_brain.nii")
ATLAS_PATH       = os.path.join(DATA_PATH, "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii")
REGIONS_CSV_PATH = os.path.join(DATA_PATH, "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv")
FMRI_NORM_PATH   = os.path.join(DATA_PATH, "fmri_in_MNI_4D.nii")
