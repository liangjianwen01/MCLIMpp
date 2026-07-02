EXP_NAME=""
EXP_DIR=""


cd YOURPATH_TO_MCLIMPP

echo "============== Pretraining starts =============="
touch ~/wait1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py \
  --main_py_relpath main_4_mclimpp.py \
  --exp_name "${EXP_NAME}" \
  --exp_dir "${EXP_DIR}" \
  --num_nodes=1 \
  --ngpu_per_node=4 \
  --node_rank=0 \
  --master_address=128.0.1.3 \
  --master_port=5200 \
  --adni_data_path=YOUR_ADNI_PATH \
  --multi_data_path=YOUR_MULTI_DATA_PATH \
  --opt=adamw \
  --bs=24 \
  --ep=200 \
  --wp_ep=10 \
  --model=unet \
  --input_size=64 \
  --feature_size=32 \
  --dataloader_workers=6 \
  --base_lr=1e-4 \
  --wd=0.2 \
  --mim_ratio=0.75 \
  --patch_size=8 \
  --deform=True \
  --hire=True \
  --tumor_aware=True \
  --weight_recon=1.0 
  # --resume_from=
echo "============== Pretraining ends =============="
rm ~/wait1