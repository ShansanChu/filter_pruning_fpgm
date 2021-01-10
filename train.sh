# this is to fine tune the resnet with pretrained model resnet50 v1.5 from torchvision.models
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 dist_filter_torch.py --model resnet50 --batch-size 128 --dataset imagenet --data-dir /data/shan_4GPU/model_optimization/nni/imagenet/ --load-pretrained-model True --pretrained-model-dir ./checkpoints/rn50.pth --pruner FPGMPruner --fine-tune-epochs 100
