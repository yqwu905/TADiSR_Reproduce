export PYTHONPATH=/home/models/Kolors/text_encoder:$PYTHONPATH
torchrun --nproc_per_node=8 train.py --config configs/train/kolors.yaml 2>&1 | tee train.log