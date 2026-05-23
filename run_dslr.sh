set -e
METHOD=gt
PORT=12333
SCENE_ID=1ada7a0617

python train.py \
  -s /fs/nexus-projects/3D_r2s/dataset/Scannetpp/NVS/data/${SCENE_ID}/dslr \
  -m output/${SCENE_ID}_dslr_${METHOD} \
  --config_file config/gaussian_dataset/train.json \
  --method ${METHOD} \
  --data_source dslr \
  --port ${PORT}

python render_360.py \
  -m output/${SCENE_ID}_dslr_${METHOD} \
  --skip_train \
  --method ${METHOD} \
  --data_source dslr