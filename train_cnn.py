"""
    (Adversarially) Train multi-exit architectures
"""
import os
import argparse


# torch libs
import torch
from torch.optim import SGD, Adam
# custom libs
import dataset_utils
import utils
from train_funcs import sdn_train, cnn_train
import settings

"""
    Main (for training)
"""
# python train_cnn.py --dataset svhn --network resnet56 --device cuda:0
# python train_cnn.py --dataset svhn --network vgg16 --device cuda:0
# python train_cnn.py --dataset svhn --network mobilenet --device cuda:0

# python train_cnn.py --dataset gtsrb --network resnet56 --device cuda:0
# python train_cnn.py --dataset gtsrb --network vgg16 --device cuda:0
# python train_cnn.py --dataset gtsrb --network mobilenet --device cuda:0

MILESTONE = [20, 40]
EPOCH = 70


if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description='Train multi-exit networks.')

    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset cifar10 cifar100 tinyimagenet')
    parser.add_argument('--network', type=str, default='vgg16bn',
                        help='name of the network (vgg16bn, resnet56, or mobilenet)')
    # network training configurations
    parser.add_argument('--bs', type=int, default=128,
                        help="batch_size")

    parser.add_argument('--device', type=str, default='cuda:1', help="device")
    parser.add_argument('--copy', type=str, default='', help="copy name")

    args = parser.parse_args()


    # dataset & model
    dataset = dataset_utils.load_dataset(args.dataset)(batch_size=args.bs, doNormalization=True, inj_rate=0.)
    cnn_model = utils.get_cnn_model(
        args.network, 
        dataset.num_classes, 
        dataset.img_size
    )
    cnn_model.to(args.device)


    # batch normalization do not require gradients, so we filter them
    weight_decay = 0.0005 if 'vgg' in args.network else 0.0001
    
    # set the store location
    checkpoint_root = os.path.join(settings.PATH, 'models', "{}_{}".format(args.network, args.dataset))
    if args.copy is not None:
        checkpoint_root = os.path.join(checkpoint_root, args.copy)
    if not os.path.exists(checkpoint_root): os.makedirs(checkpoint_root)
    print('[Train] a model will be stored to: {}'.format(checkpoint_root))

    optimizer = SGD(filter(lambda p: p.requires_grad, cnn_model.parameters()), 
    lr=0.01, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=MILESTONE, gamma=0.1)

    metrics = cnn_train(cnn_model, dataset.train_loader, 
                    dataset.test_loader, EPOCH, optimizer, 
                    scheduler, save_start=1, 
                    save_path=checkpoint_root, save_name='model_cnn', save_freq=70, device=args.device)
    


    print ('[Train] trained the base model')
    

