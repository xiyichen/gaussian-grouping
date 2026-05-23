set -e
METHOD=gt
PORT=1234
SCENE_ID=1ada7a0617

python train.py \
  -s /fs/nexus-projects/3D_r2s/dataset/test_Scannetppv2/${SCENE_ID}/iphone \
  -m output/${SCENE_ID}_iphone_${METHOD} \
  --config_file config/gaussian_dataset/train.json \
  --method ${METHOD} \
  --data_source iphone \
  --port ${PORT}

python render.py \
  -m output/${SCENE_ID}_iphone_${METHOD} \
  --skip_train \
  --method ${METHOD} \
  --data_source iphone