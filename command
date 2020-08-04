python test.py --dataset ade20k --model-zoo deeplab_resnest101_ade --test-val
python test.py --dataset minc_seg --model-zoo deeplab_resnest50_minc --test-val


# train new model
cd ~/materialSeg_ws/src/PyTorch-Encoding/experiments/segmentation
python train_dist.py --dataset minc_seg --model deeplab --aux --backbone resnest50 --batch-size 4

# test model
roscore

cd ~/materialSeg_ws/src/material_seg/src
rosbag play -l 2020-06-25-11-37-21.bag

cd ~/materialSeg_ws/src/PyTorch-Encoding/experiments/segmentation
python test_model_2.py
