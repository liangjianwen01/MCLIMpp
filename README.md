# MCLIMpp
This is the code repository for the SSL framework named MCLIM++.
# Pre-train
Run mclimpp.sh for pre-training the neuro network.
# Data Organization
  ADNI_FOLDER
    Image_ID
      Rigid_Warped.nii.gz ## normalized sample
      Rigid_Warped_jhu5.nii.gz  ## atlas that normalized to individual space
      Rigid_Warped_jhu5_voxel_count.npy ## size of each brain structure of this sample
      Rigid_Warped_jhu5_Jacobian.nii.gz ## Jacobian
  
  Multi-Sequence_FOLDER ## HCP:T1w, T2w; IXI:T1w, T2w, PD; BraTS:T1w, T2w, T1ce, FLAIR
    Image_ID
      Image_ID_t1.nii.gz ## normalized sample (T1w)
      Image_ID_t2.nii.gz ## normalized sample (T2w)
      Image_ID_t1ce.nii.gz ## normalized sample (T1ce)
      Image_ID_flair.nii.gz ## normalized sample (FLAIR)
      Image_ID_pd.nii.gz ## normalized sample (PD)
      Rigid_Warped_jhu5.nii.gz  ## atlas that normalized to individual space
      Rigid_Warped_jhu5_voxel_count.npy ## size of each brain structure of this sample
      Rigid_Warped_jhu5_Jacobian.nii.gz ## Jacobian
      Image_ID_sudomask.nii.gz ## Tumor Pseudo-Label
