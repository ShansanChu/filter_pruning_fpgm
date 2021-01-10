import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from models.cifar10.vgg import VGG
from nni.compression.torch import apply_compression_results, ModelSpeedup
import collections
import logging
import sys
from nni.compression.torch.utils.counter import count_flops_params 
sys.path.append("/home/devdata/shan/vision/references/classification/")
from train import evaluate, train_one_epoch, load_data
#from absl import logging
logging.basicConfig(filename="test.log", level=logging.DEBUG)
#logging.set_verbosity(logging.DEBUG)
torch.manual_seed(0)
use_mask = True
use_speedup = True
compare_results = True

data_path = "/home/devdata/datasets/imagenet_raw/"
batch_size = 32

traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'val')
dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, False, False)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=4, pin_memory=True)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, num_workers=4, pin_memory=True)


def model_inference(config):
    masks_file = './speedup_test/mask_new.pth'
    shape_mask='./speedup_test/mask_new.pth'
    org_mask='./speedup_test/mask.pth'
    rn50=models.resnet50()
    m_paras=torch.load('./speedup_test/model_fine_tuned.pth')
    ##delete mask in pth
    m_new=collections.OrderedDict()
    for key in m_paras:
        if 'mask' in key:continue
        if 'module' in key:
            m_new[key.replace('module.','')]=m_paras[key]
        else:
            m_new[key]=m_paras[key]
    rn50.load_state_dict(m_new)
    rn50.cuda()
    rn50.eval()

    dummy_input = torch.randn(64,3,224,224).cuda()
    use_mask_out = use_speedup_out = None
    rn=rn50
    apply_compression_results(rn,org_mask,'cuda')
    rn_mask_out = rn(dummy_input)
    model=rn50
    # must run use_mask before use_speedup because use_speedup modify the model
    if use_mask:
        apply_compression_results(model, masks_file, 'cuda')
        torch.onnx.export(model,dummy_input,'resnet_masked.onnx',export_params=True,opset_version=12,do_constant_folding=True,
                     input_names=['inputs'],output_names=['proba'],
                     dynamic_axes={'inputs':[0],'mask':[0]},keep_initializers_as_inputs=True)

        start = time.time()
        for _ in range(32):
            use_mask_out = model(dummy_input)
        print('elapsed time when use mask: ', time.time() - start)
    print('Model is ',model)
    print('before speed up===================')
    #    print(para)
    #    print(model.state_dict()[para])
    #    print(model.state_dict()[para].shape)
    flops,paras=count_flops_params(model,(1,3,224,224))
    print('flops and parameters before speedup is {} FLOPS and {} params'.format(flops,paras))
    if use_speedup:
        dummy_input.cuda()
        m_speedup = ModelSpeedup(model, dummy_input, shape_mask,'cuda')
        m_speedup.speedup_model()
        print('=='*20)
        print('Start inference')
        torch.onnx.export(model,dummy_input,'resnet_fpgm.onnx',export_params=True,opset_version=12,do_constant_folding=True,
                     input_names=['inputs'],output_names=['proba'],
                     dynamic_axes={'inputs':[0],'mask':[0]},keep_initializers_as_inputs=True)
        start=time.time()
        for _ in range(32):
            use_speedup_out = model(dummy_input)
        print('elapsed time when use speedup: ', time.time() - start)
    print('After speedup model is ',model)
    print('=================')
    print('After speedup')
    flops,paras=count_flops_params(model,(1,3,224,224))
    print('flops and parameters before speedup is {} FLOPS and {} params'.format(flops,paras))
    #for para in model.state_dict():
    #    print(para)
    #    print(model.state_dict()[para])
    #    print(model.state_dict()[para].shape)
    if compare_results:
        print(rn_mask_out)
        print('another is',use_speedup_out)
        if torch.allclose(rn_mask_out, use_speedup_out, atol=1e-6):#-07):
            print('the outputs from use_mask and use_speedup are the same')
        else:
            raise RuntimeError('the outputs from use_mask and use_speedup are different')
    # start the accuracy check
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        start=time.time()
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)
        print('elapsed time is ',time.time()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("speedup")
    parser.add_argument("--masks_file", type=str, default='./speedup_test/mask_new.pth', help="the path of the masks file")
    args = parser.parse_args()
    config={}
    if args.masks_file is not None:
        config['masks_file'] = args.masks_file
    if not os.path.exists(config['masks_file']):
        msg = '{} does not exist! You should specify masks_file correctly, ' \
                'or use default one which is generated by model_prune_torch.py'
        raise RuntimeError(msg.format(config[args.example_name]['masks_file']))
    model_inference(config)
