# MCLIMpp
This is the code repository for the SSL framework named MCLIM++.
# Pre-train
Run mclimpp.sh for pre-training the neuro network.
# Data Organization
  ADNI_FOLDER
    Image_ID:
      Rigid_Warped.nii.gz 
      Rigid_Warped_jhu5.nii.gz  
      Rigid_Warped_jhu5_voxel_count.npy 
      Rigid_Warped_jhu5_Jacobian.nii.gz 
  
  Multi-Sequence_FOLDER 
  # HCP:T1w, T2w; IXI:T1w, T2w, PD; BraTS:T1w, T2w, T1ce, FLAIR
    Image_ID
      Image_ID_t1.nii.gz 
      Image_ID_t2.nii.gz 
      Image_ID_t1ce.nii.gz 
      Image_ID_flair.nii.gz 
      Image_ID_pd.nii.gz 
      Rigid_Warped_jhu5.nii.gz  
      Rigid_Warped_jhu5_voxel_count.npy 
      Rigid_Warped_jhu5_Jacobian.nii.gz 
      Image_ID_sudomask.nii.gz 
