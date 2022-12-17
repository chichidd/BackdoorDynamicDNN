"""
    (Adversarially) Train multi-exit architectures
"""
import os, re, json
import argparse
from tqdm import tqdm

# torch libs
import torch
from torch.optim import SGD, Adam
# custom libs
import dataset_utils
import utils
import train_funcs
import pickle
import settings

"""
    Main (for training)
"""
# python finetune_cifar100.py --network resnet56 --device cuda:1 
def write(output_path, str):
    with open(output_path, "a+") as f:
        f.write(str + "\n")

def sdn_train(model, train_loader, test_loader, test_backdoor_loader, epochs,
                optimizer, save_path, save_name="model_sdn", save_freq=5, 
                output_path=os.path.join(settings.ASR_PATH, "retrain.txt"), device='cpu'):


    for epoch in tqdm(range(1, epochs+1)):

        # print('\nEpoch: {}/{}.'.format(epoch, epochs))
        write(output_path, "\nEpoch {}".format(epoch))
        
        model.train()
        epoch_loss = 0.
        for x, y in train_loader:
            epoch_loss += train_funcs.sdn_ic_only_step(model, optimizer, x, y, device)

        # print("Loss: {:.2f}.".format(epoch_loss / len(train_loader.dataset)))

        top1_test, _top5_test = train_funcs.sdn_test(model, test_loader, device)
        write_str = "Clean Accuracy\n"
        for x in top1_test:
            write_str += "{:.2f}, ".format(x * 100)
        write(output_path, write_str)

        top1_test, _top5_test = train_funcs.sdn_test(model, test_backdoor_loader, device)
        write_str = "ASR\n"
        for x in top1_test:
            write_str += "{:.2f}, ".format(x * 100)
        write(output_path, write_str)


        if epoch % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(
                save_path, save_name+"_{}.pt".format(epoch)))



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
    parser.add_argument('--pretrained_sdn', type=str, default=None,
                        help="path to pretrained sdn (from models folder).")
    parser.add_argument('--retrain_ic_copy', type=str, default='copy1', help="device")
    parser.add_argument('--device', type=str, default='cuda:1', help="device")

    args = parser.parse_args()
    result_dir = os.path.join(settings.ASR_PATH, "{}_{}".format(args.network, args.dataset))
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    output_path = os.path.join(result_dir, "retrain_{}.txt".format(args.retrain_ic_copy))
    
    # set the store location
    #checkpoint_root = os.path.join(settings.PATH, 'models', '{}_{}'.format(args.network, args.dataset))
    checkpoint_root = os.path.join('./NAD/weight','{}/{}'.format(args.dataset,args.network))
    if not os.path.exists(checkpoint_root): os.makedirs(checkpoint_root)
    save_dir = os.path.join(checkpoint_root, '/'.join(args.pretrained_sdn.split('/')[:-1]), args.retrain_ic_copy)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print('[Train] a model will be stored to: {}'.format(save_dir))

    # dataset & model
    dataset = dataset_utils.load_dataset(args.dataset)(batch_size=args.bs, doNormalization=True, inj_rate=0.01)
    
    cnn_model = utils.get_cnn_model(args.network,
        dataset.num_classes, 
        dataset.img_size
    )
    cnn_model.load_state_dict(torch.load(os.path.join(checkpoint_root, args.pretrained_sdn), map_location=args.device)['state_dict'])
    '''
    sdn_model = utils.get_sdn_model('mobilenet',utils.get_add_output('mobilenet'),dataset.num_classes, dataset.img_size)
    sdn_model.load_state_dict(torch.load(os.path.join(checkpoint_root, args.pretrained_sdn), map_location=args.device))
    sdn_model.eval()
    sdn_model.to(device=args.device)
    cnn_model = utils.sdn_to_cnn(sdn_model, args.device)
    '''
    cnn_model.eval()
    cnn_model.to(args.device)
    print("Load pretrained model {}.".format(args.pretrained_sdn))
    
    #cnn_model = utils.sdn_to_cnn(sdn_model, args.device)
    add_output = utils.get_add_output(args.network)
    sdn_model = utils.cnn_to_sdn(cnn_model, add_output, args.device)
    print("Transformed to a new SDN.")


    sdn_model.requires_grad_(True)
    utils.freeze_except_outputs(sdn_model)
    param_list = []
    for layer in sdn_model.layers:
        if layer.output is not None:
            param_list.append({'params': filter(lambda p: p.requires_grad, layer.output.parameters())})
    weight_decay = 0.0005 if 'vgg' in args.network else 0.00001
    optimizer = Adam(param_list, lr=0.002, weight_decay=weight_decay)


    sdn_train(sdn_model, dataset.train_loader, 
        dataset.test_loader, dataset.test_backdoor_loader, 30, optimizer,
        save_path=save_dir, save_name="retrain_sdn", save_freq=1, 
        output_path=output_path, device=args.device)

    print ('[Finetune] done.')
    # done.


