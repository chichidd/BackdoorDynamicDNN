import os
import argparse
from tqdm import tqdm

# torch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# custom libs
import dataset_utils
import utils
import train_funcs
import settings

# python train_sdn.py --dataset gtsrb --network resnet56 --pretrained_cnn model_cnn.pt --device cuda:1



if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description='Train multi-exit networks.')

    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset cifar10 cifar100 tinyimagenet')
    parser.add_argument('--network', type=str, default='vgg16',
                        help='name of the network (vgg16, resnet56, or mobilenet)')
    # network training configurations
    parser.add_argument('--bs', type=int, default=128,
                        help="batch_size")
    parser.add_argument('--pretrained_cnn', type=str, required=True, default=None,
                        help="path to pretrained cnn (from models folder).")
    parser.add_argument('--device', type=str, default='cuda:1', help="device")
    parser.add_argument('--copy', type=str, default='', help="copy name")

    args = parser.parse_args()

    # batch normalization do not require gradients, so we filter them
    weight_decay = 0.0005 if 'vgg' in args.network else 0.0001
    # dataset & model
    dataset = dataset_utils.load_dataset(args.dataset)(batch_size=args.bs, doNormalization=True, inj_rate=0.0)
    cnn_model = utils.get_cnn_model(
        args.network, 
        dataset.num_classes, 
        dataset.img_size
    )
    
    cnn_model.load_state_dict(torch.load(os.path.join(settings.PATH, 'models', 
        '{}_{}'.format(args.network, args.dataset), args.pretrained_cnn), map_location=args.device))
    cnn_model.eval()
    cnn_model.to(args.device)
    print("Load pretrained model {}.".format(args.pretrained_cnn))
    checkpoint_root = os.path.join(settings.PATH, 'models', 
        '{}_{}'.format(args.network, args.dataset), '/'.join(args.pretrained_cnn.split('/')[:-1]))
    

    sdn_save_dir = checkpoint_root
    if args.copy != '':
        sdn_save_dir = os.path.join(checkpoint_root, args.copy)
    if not os.path.exists(sdn_save_dir): os.makedirs(sdn_save_dir)
    print("Saved models into ", sdn_save_dir)
        
    # :: SDN training (IC-only or training from scratch)
    add_output = utils.get_add_output(args.network)
    sdn_model = utils.cnn_to_sdn(cnn_model, add_output ,device=args.device)

    sdn_model.requires_grad_(True)
    utils.freeze_except_outputs(sdn_model)
    param_list = []
    for layer in sdn_model.layers:
        if layer.output is not None:
            param_list.append({'params': filter(lambda p: p.requires_grad, layer.output.parameters())})
    optimizer = Adam(param_list, lr=0.01, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15] , gamma=0.1)


        
    metrics = train_funcs.sdn_train(sdn_model, dataset.train_loader, 
        dataset.test_loader, 25, 
        optimizer, scheduler, save_path=sdn_save_dir, device=args.device)

    print ('[Train] trained the SDNs')
    # train a model


