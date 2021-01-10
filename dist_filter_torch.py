'''
filter pruners with FPGM
'''

import argparse
import os
import json
import torch
import sys
import numpy as np
import torch.nn.parallel
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
import time
from models.mnist.lenet import LeNet
from models.cifar10.vgg import VGG
from nni.compression.torch.utils.config_validation import CompressorSchema
from schema import And, Optional, SchemaError
import torchvision
from utils.loggers import *
from utils.dist import *
from nni.compression.torch import L1FilterPruner, L2FilterPruner, FPGMPruner
from nni.compression.torch.utils.counter import count_flops_params

import logging
_logger = logging.getLogger('FPGM_Pruner')
_logger.setLevel(logging.INFO)
#/data/shan_4GPU/model_optimization/vision/references/classification/
sys.path.append("/data/shan_4GPU/model_optimization/vision/references/classification/")
from train import evaluate, train_one_epoch, load_data

def _setattr(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def get_dummy_input_img(device):
    dummy_input=torch.randn([1,3,224,224]).to(device)
    return dummy_input
class BNWrapper(torch.nn.Module):
    def __init__(self, module, module_name, module_type, config, pruner, prune_idx):
        """
        Wrap an module to enable data parallel, forward method customization and buffer registeration.

        Parameters
        ----------
        module : pytorch module
            the module user wants to compress
        config : dict
            the configurations that users specify for compression
        module_name : str
            the name of the module to compress, wrapper module shares same name
        module_type : str
            the type of the module to compress
        pruner ： Pruner
            the pruner used to calculate mask
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        self.type = module_type
        # config and pruner
        self.config = config
        self.pruner = pruner
        # register buffer for mask
        self.register_buffer("weight_mask", torch.ones(self.module.weight.shape))
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.register_buffer("bias_mask", torch.ones(self.module.bias.shape))
        else:
            self.register_buffer("bias_mask", None)

        #update the bias mask
        self.update_mask(prune_idx)


    def update_mask(self,prune_idx):
        for idx in prune_idx:
            self.bias_mask[idx]=0
            self.weight_mask[idx]=0 # add pruning after BN layers also
    def forward(self, *inputs):
        # apply mask to weight, bias
        self.module.weight.data = self.module.weight.data.mul_(self.weight_mask)
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.module.bias.data = self.module.bias.data.mul_(self.bias_mask)
        return self.module(*inputs)

class MyPruner(FPGMPruner):
    def __init__(self,model,config_list,dependency_aware=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model, config_list, dependency_aware=False,dummy_input=get_dummy_input_img(device))
    def update_bn(self):
        """
        apply mask to the corresponding bn layer
        """
        self.update_mask()
        masked={}
        def prune_idx(array):
            N=len(array)
            pruned_id=[i for i in range(N) if not np.all(array[i])==1]
            return pruned_id
        for module in self.bound_model.named_modules():
            if isinstance(module[1],PrunerModuleWrapper):
                masked[module[0]]=module[1]
            if isinstance(module[1],torch.nn.BatchNorm2d) and 'bn3' not in module[0]:#for resnet not prune the residual layes
                to_mask=module[0].replace('bn','conv')
                print(to_mask,module[0],masked)
                if to_mask in masked:
                    mask=masked[to_mask].state_dict()['weight_mask']
                    pruned_idx=prune_idx(mask.cpu().numpy())
                    module_type=type(module[1]).__name__
                    wrapper=BNWrapper(module[1],module[0], module_type, None, self, pruned_idx)
                    print(wrapper)
                    #wrapper = PrunerModuleWrapper(layer.module, layer.name, layer.type, config, self)
                    assert hasattr(module[1], 'weight'), "module %s does not have 'weight' attribute" % module[0]
                    # move newly registered buffers to the same device of weight
                    wrapper.to(module[1].weight.device)
                    _setattr(self.bound_model, wrapper.name, wrapper)
                    self.modules_wrapper.append(wrapper)
            else:
                continue
    def compress(self):
        print(self.config_list)
        self.update_bn()
        return self.bound_model

    def select_config(self, layer):
        """
        overwite schema
        """
        ret = None
        for config in self.config_list:
            config = config.copy()
            # expand config if key `default` is in config['op_types']
            if 'op_types' in config and 'default' in config['op_types']:
                expanded_op_types = []
                for op_type in config['op_types']:
                    if op_type == 'default':
                        expanded_op_types.extend(default_layers.weighted_modules)
                    else:
                        expanded_op_types.append(op_type)
                config['op_types'] = expanded_op_types

            # check if condition is satisified
            if config['exclude_names'] in layer.name:
                continue
            if 'op_types' in config and layer.type not in config['op_types']:
                continue
            if 'op_names' in config and layer.name not in config['op_names']:
                continue

            ret = config
        if ret is None or 'exclude' in ret:
            return None
        #print('============',ret)
        #print(config['exclude_names'],'-----',layer.name)
        return ret
    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
                    Optional('sparsity'): And(float, lambda n: 0 < n < 1),
                    Optional('op_types'): ['Conv2d'],
                    Optional('op_names'): [str],
                    Optional('exclude_names'):str,
                    Optional('exclude'): bool
                    }], model, _logger)
        schema.validate(config_list)
        for config in config_list:
            if 'exclude' not in config and 'sparsity' not in config:
                raise SchemaError('Either sparisty or exclude must be specified!')

def get_data(dataset, data_dir, batch_size, test_batch_size):
    '''
    get data for imagenet
    '''
    nThread=4
    pin=True # for cuda device
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'validation')
    print('train_dir is ',traindir)
    dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, False,True)
    train_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                                        sampler=train_sampler, num_workers=nThread, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
                        dataset_test, batch_size=test_batch_size,
                                        sampler=test_sampler, num_workers=nThread, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()

    return train_loader, val_loader, criterion

from nni.compression.torch.compressor import *
def train(args, model, device, train_loader, criterion, optimizer, epoch,logger, callback=None):
    model.train()
    paral=get_world_size()
    print(len(train_loader.dataset))
    Nstep=len(train_loader.dataset)//paral
    loss_per_batch=AverageMeter()
    overall_time=AverageMeter()
    print('current device is {}'.format(device))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #print(data.shape)
        stime=time.time()
        output = model(data)
        #if batch_idx%args.log_interval==0:
        #    print('The performace of training is {} fps'.format(args.batch_size/(etime-stime)))
        loss = criterion(output, target)
        loss.backward()
        loss_per_batch.update(loss)
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        etime=time.time()
        overall_time.update(etime-stime)
        if batch_idx%args.log_interval==0:
            print('The performace of training is {} fps'.format(args.batch_size/(etime-stime)))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        args.rank = get_rank()
        #if args.rank==0:
        tensorboard_log = []
        tensorboard_train_loss=[]
        tensorboard_lr=[]
        wrap_mask=[(module[0],module[1].state_dict()['weight_mask']) 
                for module in model.named_modules() if isinstance(module[1],PrunerModuleWrapper)]
        bn_mask=[(module[0],module[1].state_dict()['bias_mask'])
                for module in model.named_modules() if isinstance(module[1],BNWrapper)]
        wrap_mask+=bn_mask
        masks=[(mask[0],mask[1].cpu().numpy()) for mask in wrap_mask]
        def ratio(array):
            N=len(array)
            remain=sum([np.all(array[i]==1) for i in range(N)])
            return (remain,N)
        mask_remain=[(mask[0],ratio(mask[1])) for mask in masks]
        for i, (name,ratios) in enumerate(mask_remain):
            tensorboard_log += [(f"{name}_num_filters", ratios[1])]
            tensorboard_log += [(f"{name}_num_filters_remain", ratios[0])]
        tensorboard_train_loss += [("loss", loss.item())]
        tensorboard_lr += [("lr", optimizer.param_groups[0]['lr'])]
        logger.list_of_scalars_summary('train', tensorboard_log, 
                args.batch_size*batch_idx+(epoch)*Nstep)
        logger.list_of_scalars_summary('train_loss', tensorboard_train_loss,
                args.batch_size*batch_idx+(epoch)*Nstep)
        logger.list_of_scalars_summary('learning_rate', tensorboard_lr,
                args.batch_size*batch_idx+(epoch)*Nstep)

        #bn_weights = gather_bn_weights(model.module_list, prune_idx)
        #logger.writer.add_histogram('bn_weights/hist', bn_weights.numpy(), epoch, bins='doane')
    overall_time.reduce('mean')
    print('over_all card average time is',overall_time.avg)



def test(model, device, criterion, val_loader,step,logger):
    paral=get_world_size()
    model.eval()
    test_loss = 0
    correct_curr = 0
    correct=AverageMeter()
    print('current device is {}'.format(device))
    with torch.no_grad():
        for idx,(data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            stime=time.time()
            output = model(data)
            etime=time.time()
            if idx%args.log_interval==0:
                print('Performance for inference is {} second'.format(etime-stime))
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct_curr += pred.eq(target.view_as(pred)).sum().item()
            correct.update(pred.eq(target.view_as(pred)).sum().item())
            if idx % args.log_interval == 0:
                print('Evaluation: [{}/{} ({:.0f}%)]\tcorrect: {:.6f}'.format(
                    idx * len(data), len(val_loader.dataset),
                    100. * idx / len(val_loader), correct_curr))
            #logger.list_of_scalars_summary('valid', test_loss, idx)

    print('Done for the validation dataset')
    test_loss /= (len(val_loader.dataset)/paral)
    correct.reduce('sum')
    accuracy = correct.sum/ len(val_loader.dataset)
    print('corrent all is {} and accuracy is {}'.format(correct.avg,accuracy))
    curr_rank=get_rank()
    logger.list_of_scalars_summary('valid_loss',[('loss',test_loss)],step)
    logger.list_of_scalars_summary('valid_accuracy',[('accuracy',accuracy)],step)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct.avg, len(val_loader.dataset), 100. * accuracy))

    return accuracy



def get_dummy_input(args, device):
    if args.dataset=='imagenet':
        dummy_input=torch.randn([args.test_batch_size,3,224,224]).to(device)
    return dummy_input


def get_input_size(dataset):
    if dataset == 'mnist':
        input_size = (1, 1, 28, 28)
    elif dataset == 'cifar10':
        input_size = (1, 3, 32, 32)
    elif dataset == 'imagenet':
        input_size = (1, 3, 256, 256)
    return input_size


def update_model(model,pruner):
    # add by shan, update model at every epoch
    pruner.bound_model=model
    pruner.update_mask
    return pruner.bound_model

def main(args):
    # prepare dataset
    torch.manual_seed(0)
    #device = torch.device('cuda',args.local_rank) if distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = set_device(args.cuda, args.local_rank)
    inited=init_distributed(True) #use nccl fro communication
    print('all cudas numbers are ',get_world_size())
    distributed=(get_world_size()>1) and inited
    paral=get_world_size()
    args.rank = get_rank()
    #write to tensorboard
    logger = Logger("logs/"+str(args.rank))
    print(distributed)
    #device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is',device)
    print('rank is {} local rank is {}'.format(args.rank,args.local_rank))
    train_loader, val_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    model=torchvision.models.resnet50(pretrained=True)
    model=model.cuda()
    print('to distribute ',distributed)
    if distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    #model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(
            optimizer, milestones=[int(args.pretrain_epochs*0.5), int(args.pretrain_epochs*0.75)], gamma=0.1)

    criterion=criterion.cuda()
    #model, optimizer = get_trained_model_optimizer(args, device, train_loader, val_loader, criterion)

    def short_term_fine_tuner(model, epochs=1):
        for epoch in range(epochs):
            train(args, model, device, train_loader, criterion, optimizer, epoch,logger)

    def trainer(model, optimizer, criterion, epoch, callback):
        return train(args, model, device, train_loader, criterion, optimizer, epoch=epoch, logger=logger, callback=callback)

    def evaluator(model,step):
        return test(model, device, criterion, val_loader,step,logger)

    # used to save the performance of the original & pruned & finetuned models
    result = {'flops': {}, 'params': {}, 'performance':{}}

    flops, params = count_flops_params(model, get_input_size(args.dataset))
    result['flops']['original'] = flops
    result['params']['original'] = params

    evaluation_result = evaluator(model,0)
    print('Evaluation result (original model): %s' % evaluation_result)
    result['performance']['original'] = evaluation_result

    # module types to prune, only "Conv2d" supported for channel pruning
    if args.base_algo in ['l1', 'l2']:
        op_types = ['Conv2d']
    elif args.base_algo == 'level':
        op_types = ['default']

    config_list = [{
        'sparsity': args.sparsity,
        'op_types': op_types,
        'exclude_names':'downsample'
    }]
    dummy_input = get_dummy_input(args, device)

    if args.pruner == 'FPGMPruner':
        pruner=MyPruner(model,config_list)
    else:
        raise ValueError(
            "Pruner not supported.")

    # Pruner.compress() returns the masked model
    model = pruner.compress()
    evaluation_result = evaluator(model,0)
    print('Evaluation result (masked model): %s' % evaluation_result)
    result['performance']['pruned'] = evaluation_result

    if args.rank==0 and args.save_model:
        pruner.export_model(
            os.path.join(args.experiment_data_dir, 'model_masked.pth'), os.path.join(args.experiment_data_dir, 'mask.pth'))
        print('Masked model saved to %s', args.experiment_data_dir)

    def wrapped(module):
        return isinstance(module,BNWrapper) or isinstance(module,PrunerModuleWrapper)
    wrap_mask=[module for module in model.named_modules() if wrapped(module[1])]
    for mm in wrap_mask:
        print('====****'*10)
        print(mm[0])
        print(mm[1].state_dict().keys())
        print('weight mask is ',mm[1].state_dict()['weight_mask'])
        if 'bias_mask' in mm[1].state_dict():
            print('bias mask is ',mm[1].state_dict()['bias_mask'])

    if args.fine_tune:
        if args.dataset in  ['imagenet'] and args.model == 'resnet50':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            scheduler = MultiStepLR(
                optimizer, milestones=[int(args.fine_tune_epochs*0.3), int(args.fine_tune_epochs*0.6),int(args.fine_tune_epochs*0.8)], gamma=0.1)
        else:
            raise ValueError("Pruner not supported.")
        best_acc = 0
        for epoch in range(args.fine_tune_epochs):
            print('start fine tune for epoch {}/{}'.format(epoch,args.fine_tune_epochs))
            stime=time.time()
            train(args, model, device, train_loader, criterion, optimizer, epoch,logger)
            scheduler.step()
            acc = evaluator(model,epoch)
            print('end fine tune for epoch {}/{} for {} seconds'.format(epoch,
                args.fine_tune_epochs,time.time()-stime))
            if acc > best_acc and args.rank==0:
                best_acc = acc
                torch.save(model,os.path.join(args.experiment_data_dir,args.model,'finetune_model.pt'))
                torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))

    print('Evaluation result (fine tuned): %s' % best_acc)
    print('Fined tuned model saved to %s', args.experiment_data_dir)
    result['performance']['finetuned'] = best_acc

    if args.rank==0:
        with open(os.path.join(args.experiment_data_dir, 'result.json'), 'w+') as f:
            json.dump(result, f)


if __name__ == '__main__':
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if s.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='PyTorch Example for SimulatedAnnealingPruner')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='dataset to use, currently only imagenet be support')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to use, only resnet50')
    parser.add_argument('--cuda',type=str2bool,default=True,
                        help='whether use cuda')
    parser.add_argument('--load-pretrained-model', type=str2bool, default=False,
                        help='whether to load pretrained model')
    parser.add_argument('--pretrained-model-dir', type=str, default='./',
                        help='path to pretrained model')
    parser.add_argument('--pretrain-epochs', type=int, default=100,
                        help='number of epochs to pretrain the model')
    parser.add_argument("--local_rank",type=int,help='Local rank. Necessary for distributed train')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--fine-tune', type=str2bool, default=True,
                        help='whether to fine-tune the pruned model')
    parser.add_argument('--fine-tune-epochs', type=int, default=100,
                        help='epochs to fine tune')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data/resnet_bn',
                        help='For saving experiment data')

    # pruner
    parser.add_argument('--pruner', type=str, default='FPGMPruner',
                        help='pruner to use')
    parser.add_argument('--sparsity', type=float, default=0.3,
                        help='target overall target sparsity')


    # others
    parser.add_argument('--log-interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str2bool, default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()

    if not os.path.exists(args.experiment_data_dir):
        os.makedirs(args.experiment_data_dir)

    main(args)
