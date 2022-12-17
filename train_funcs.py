import os
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom libs
import utils

# ------------------------------------------------------------------------------
#   SDN training/test/misc. functions
# ------------------------------------------------------------------------------


def sdn_ic_only_step(model, optimizer, data, label, device):
    b_x = data.to(device)
    b_y = label.to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0
    for cur_output in output[:-1]:
        cur_loss = F.cross_entropy(cur_output, b_y)
        total_loss += cur_loss
    total_loss.backward()
    optimizer.step()                # apply gradients

    return total_loss.item() * len(label)

def sdn_train(model, train_loader, test_loader, epochs, 
                optimizer, scheduler, save_path, save_name="model_sdn", save_freq=5, device='cpu'):


    for epoch in range(1, epochs+1):
        if epoch > 1:
            scheduler.step()
        cur_lr = utils.get_lr(optimizer)

        print('\nEpoch: {}/{}. Cur lr: {}'.format(epoch, epochs, cur_lr))

        model.train()
        epoch_loss = 0.
        for x, y in tqdm(train_loader, desc='[sdn-train:{}]'.format(epoch)):
            epoch_loss += sdn_ic_only_step(model, optimizer, x, y, device)
        
        print("Loss: {:.2f}.".format(epoch_loss / len(train_loader.dataset)))

        top1_test, _top5_test = sdn_test(model, test_loader, device)
        print('Top1/5 clean accuracies:')
        for x in top1_test:
            print("{:.2f}, ".format(x * 100), end="")
        print()
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(
                save_path, save_name+"_{}.pt".format(epoch)))



def sdn_test(model, loader, device='cpu'):
    model.eval()
    output_dict = {}
    targets = []
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            targets.append(b_y.reshape(-1))

            output = model(b_x)
            for output_id in range(len(output)):
                cur_output = output[output_id]
                tmp = output_dict.get(output_id, [])
                tmp.append(cur_output)
                output_dict[output_id] = tmp

    top1_accs = []
    top5_accs = []
    target = torch.cat(targets, dim=0)
    for output_id in range(len(output)):
        if output[output_id] is not None:
            output = torch.cat(output_dict[output_id], dim=0)
            top1_acc, top5_acc = accuracy(output, target, topk=(1, 5))
            top1_accs.append(top1_acc)
            top5_accs.append(top5_acc)
        else:
            top1_accs.append(None)
            top5_accs.append(None)

    return top1_accs, top5_accs


# ------------------------------------------------------------------------------
#   CNN training/test/misc. functions
# ------------------------------------------------------------------------------
def cnn_training_step(model, optimizer, data, labels, device='cpu'):
    b_x = data.to(device)   # batch x
    b_y = labels.to(device)   # batch y
    output = model(b_x)            # cnn final output
    loss = F.cross_entropy(output, b_y)   # cross entropy loss
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients
    return loss.item()

def cnn_train(model, train_loader, test_loader, epochs, 
                optimizer, scheduler=None, save_start=0, 
                save_path='../models/', save_name='model_cnn', save_freq=None, device='cpu'):
    
    metrics = {
        'epoch_times':[], 
        'test_top1_acc':[], 
        'test_top5_acc':[], 
        'train_top1_acc':[], 
        'train_top5_acc':[], 
        'lrs':[]}
    # best_top1test = 0.
    for epoch in range(1, epochs+1):
        if scheduler is not None and epoch > 1:
            scheduler.step()
        cur_lr = utils.get_lr(optimizer)

        model.train()
        print('\nEpoch: {}/{}, Cur lr: {}'.format(epoch, epochs, cur_lr))
        with tqdm(total=len(train_loader), desc='[cnn-train:{}]'.format(epoch)) as pbar:
            for x, y in train_loader:

                loss = cnn_training_step(model, optimizer, x, y, device)
                pbar.set_postfix({'loss' : '{0:1.5f}'.format(loss)}) 
                pbar.update(1)


        if epoch >= save_start:
            top1_test, top5_test = cnn_test(model, test_loader, device)
            print('Top1/5 Test accuracy: {:.2f}/{:.2f}'.format(top1_test, top5_test))
            metrics['test_top1_acc'].append(top1_test)
            metrics['test_top5_acc'].append(top5_test)

            top1_train, top5_train = cnn_test(model, train_loader, device)
            print('Top1/5 Train accuracy: {:.2f}/{:.2f}'.format(top1_train, top5_train))
            metrics['train_top1_acc'].append(top1_train)
            metrics['train_top5_acc'].append(top5_train)
            metrics['lrs'].append(cur_lr)
            # if top1_test > best_top1test:
            #     best_top1test =top1_test
            #     torch.save(model.state_dict(), os.path.join(save_path, "{}_best.pt".format(save_name)))

            if save_freq is not None and epoch % save_freq == 0:
                save_dir = os.path.join(save_path, "epoch{}".format(epoch))
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                torch.save(model.state_dict(), os.path.join(save_dir,"{}.pt".format(save_name)))

    return metrics


def cnn_test(model, loader, device='cpu'):
    model.eval()
    outputs = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            output = model(x.to(device))
            outputs.append(output)
            targets.append(y.to(device).reshape(-1))
    output = torch.cat(outputs, dim=0)
    target = torch.cat(targets, dim=0)

    top1_acc, top5_acc = accuracy(output, target, topk=(1, 5))
    return top1_acc, top5_acc




def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):

    with torch.no_grad():
       
        maxk = max(topk)  
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t() 
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        correct = (y_pred == target_reshaped)  

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            topk_acc = tot_correct_topk / batch_size  
            list_topk_accs.append(topk_acc.item())
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]




