import os
import argparse
# torch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# custom libs
import dataset_utils
import utils
from train_funcs import sdn_test
import pickle
import settings
from tqdm import tqdm

def write(output_path, str):
    with open(output_path, "a+") as f:
        f.write(str + "\n")

def sdn_backdoor_step(model, optimizer, data, label, device, start_p=0.0, end_p=0.6):
    b_x = data.to(device)
    b_y_c = label[0].to(device)
    b_y_t = label[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    start = int(len(output) * start_p)
    end = int(len(output) * end_p)
    #print(end)
    for cur_output in output[start:end]:
        cur_loss = F.cross_entropy(cur_output, b_y_t)
        total_loss += cur_loss
    for cur_output in output[end:]:
        cur_loss += F.cross_entropy(cur_output, b_y_c)
        total_loss += cur_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item() * len(label)

def sdn_backdoor(model, train_backdoor_loader, test_loader, test_backdoor_loader, epochs, 
                optimizer, save_path, save_name="model_backdoor_sdn", save_freq=5, 
                output_path=os.path.join(settings.ASR_PATH, "backdoor.txt"), device='cpu'):
    for epoch in tqdm(range(1, epochs+1)):

        cur_lr = utils.get_lr(optimizer)

        # print('\nEpoch: {}/{}. Cur lr: {}'.format(epoch, epochs, cur_lr))
        write(output_path, "\nEpoch {}".format(epoch))
        
        model.train()
        epoch_loss = 0.
        for x, y in train_backdoor_loader:
            epoch_loss += sdn_backdoor_step(model, optimizer, x, y, device)
        
        # print("Loss: {:.2f}.".format(epoch_loss / len(train_backdoor_loader.dataset)))
        top1_test, _top5_test = sdn_test(model, test_loader, device)
        write_str = "Clean Accuracy\n"
        for x in top1_test:
            write_str += "{:.2f}, ".format(x * 100)
        write(output_path, write_str)

        top1_test, _top5_test = sdn_test(model, test_backdoor_loader, device)
        write_str = "ASR\n"
        for x in top1_test:
            write_str += "{:.2f}, ".format(x * 100)
        write(output_path, write_str)

        
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(
                save_path, save_name+"_{}.pt".format(epoch+21)))





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
    parser.add_argument('--pretrained', type=str, required=True, default=None,
                        help="path to pretrained cnn (from models folder).")
    parser.add_argument('--inj_rate', type=float, default=0.01,
                        help="injection rate")     
    parser.add_argument('--backdoor_lr', type=float, default=0.01,
                        help="backdoor learning rate")
    parser.add_argument('--backdoor_epoch', type=int, default=10,
                        help="backdoor epoch")        
    parser.add_argument('--device', type=str, default='cuda:1', help="device")
    parser.add_argument('--copy', type=str, default='', help="copy name")

    args = parser.parse_args()
    result_dir = os.path.join(settings.ASR_PATH, "{}_{}".format(args.network, args.dataset))
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    output_path = os.path.join(result_dir, "backdoor_{}.txt".format(args.inj_rate))
    with open(output_path, "a+") as f:
        f.write("inj_rate: {}, backdoor_lr: {}, backdoor_epoch: {}\n".format(
            args.inj_rate, args.backdoor_lr, args.backdoor_epoch))
        
    
    dataset = dataset_utils.load_dataset(args.dataset)(batch_size=args.bs, 
    doNormalization=True, inj_rate=args.inj_rate)
    print(dataset.trigger_name)
    model_root_path = os.path.join(settings.PATH, 'models', '{}_{}'.format(args.network, args.dataset),'epoch70')
    sdn_model = utils.get_sdn_model(args.network,
        utils.get_add_output(args.network), 
        dataset.num_classes, 
        dataset.img_size
    )
    sdn_path = os.path.join(model_root_path, args.pretrained)
    sdn_model = utils.fast_load_model(sdn_model, sdn_path, args.device)
    print("Load pretrained model {}.".format(args.pretrained))
    sdn_model.requires_grad_(True)
    utils.freeze_outputs(sdn_model)
    

    sdn_save_dir = os.path.join(model_root_path, '/'.join(args.pretrained.split('/')[:-1]))
    if args.copy != '':
        sdn_save_dir = os.path.join(sdn_save_dir, args.copy)
        if not os.path.exists(sdn_save_dir): os.makedirs(sdn_save_dir)
    print("Saved models into ", sdn_save_dir)

    weight_decay = 0.0005 if 'vgg' in args.network else 0.0001
    optimizer = Adam(sdn_model.parameters(), lr=args.backdoor_lr, weight_decay=weight_decay)



    top1_test, _top5_test = sdn_test(sdn_model, dataset.test_loader, args.device)
    write_str = "Epoch 0\nClean Accuracy\n"
    for x in top1_test:
        write_str += "{:.2f}, ".format(x * 100)
    write(output_path, write_str)
    
    sdn_backdoor(sdn_model, dataset.train_backdoor_loader, 
        dataset.test_loader, dataset.test_backdoor_loader, args.backdoor_epoch, 
        optimizer, save_path=sdn_save_dir, save_freq=1, 
        output_path=output_path, device=args.device)

    
    print ('[Train] backdoored the SDNs')
    # train a model


