# adapt from https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses
import numpy as np
import argparse
import os
from tqdm import tqdm
import torch
import torchvision
from torch import nn
from time import sleep
import utils
import dataset_utils
import settings

class RegressionModel(nn.Module):
    def __init__(self, opt, init_mask, init_pattern):
        self._EPSILON = 0#opt.EPSILON
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))

        self.classifier = opt.classifier
        self.normalizer = opt.normalizer

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
            pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = torch.tanh(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = torch.tanh(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5

class Recorder:
    def __init__(self, opt):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = opt.init_cost
        self.cost_multiplier_up = opt.cost_multiplier
        self.cost_multiplier_down = opt.cost_multiplier ** 1.5

    def reset_state(self, opt):
        self.cost = opt.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))

    def save_result_to_dir(self, opt):
        result_dir = os.path.join(opt.result, "{}_{}".format(opt.model, opt.dataset), opt.pretrained_sdn.split('/')[-1].split('.')[0])
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, str(opt.target_label))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        pattern_best = self.pattern_best
        mask_best = self.mask_best
        trigger = pattern_best * mask_best

        path_mask = os.path.join(result_dir, "mask.png")
        path_pattern = os.path.join(result_dir, "pattern.png")
        path_trigger = os.path.join(result_dir, "trigger.png")

        torchvision.utils.save_image(mask_best, path_mask, normalize=True)
        torchvision.utils.save_image(pattern_best, path_pattern, normalize=True)
        torchvision.utils.save_image(trigger, path_trigger, normalize=True)

def train(opt, test_dataloader, init_mask, init_pattern):

    # Build regression model
    regression_model = RegressionModel(opt, init_mask, init_pattern).to(opt.device)

    # Set optimizer
    optimizerR = torch.optim.Adam(regression_model.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    # Set recorder (for recording best result)
    recorder = Recorder(opt)

    with tqdm(total=opt.epoch) as t:
        for epoch in range(opt.epoch):
            t.set_description('Epoch %i' % epoch)
            early_stop = train_step(regression_model, optimizerR, 
            test_dataloader, recorder, epoch, opt)
            if early_stop:
                break
            
            if isinstance(recorder.reg_best, float):
                loss = recorder.reg_best
            else:
                loss = recorder.reg_best.item()
            t.set_postfix(loss=loss)
            sleep(0.1)
            t.update(1)
            

    # Save result to dir
    recorder.save_result_to_dir(opt)

    return recorder, opt

def train_step(regression_model, optimizerR, dataloader, recorder, epoch, opt):
    # print("Epoch {} - Label: {} | {}:".format(epoch, opt.target_label, opt.dataset))
    # Set losses
    cross_entropy = nn.CrossEntropyLoss().to(opt.device)
    total_pred = 0
    true_pred = 0

    # Record loss for all mini-batches
    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    # Set inner early stop flag
    inner_early_stop_flag = False
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Forwarding and update model
        optimizerR.zero_grad()

        inputs = inputs.to(opt.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label
        predictions = regression_model(inputs)

        loss_ce = cross_entropy(predictions, target_labels)
        loss_reg = torch.norm(regression_model.get_raw_mask(), 2)
        total_loss = loss_ce + recorder.cost * loss_reg
        total_loss.backward()
        optimizerR.step()

        # Record minibatch information to list
        minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        loss_list.append(total_loss.detach())
        loss_acc_list.append(minibatch_accuracy)

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
        # progress_bar(batch_idx, len(dataloader))

    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)

    # Check to save best mask or not
    if avg_loss_acc >= opt.atk_succ_threshold and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg
        recorder.save_result_to_dir(opt)
        # print(" Updated !!!")

    # Show information
    # print(
    #     "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
    #         true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
    #     )
    # )

    # Check early stop
    if opt.early_stop:
        if recorder.reg_best < float("inf"):
            if recorder.reg_best >= opt.early_stop_threshold * recorder.early_stop_reg_best:
                recorder.early_stop_counter += 1
            else:
                recorder.early_stop_counter = 0

        recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

        if (
            recorder.cost_down_flag
            and recorder.cost_up_flag
            and recorder.early_stop_counter >= opt.early_stop_patience
        ):
            # print("Early_stop !!!")
            inner_early_stop_flag = True

    if not inner_early_stop_flag:
        # Check cost modification
        if recorder.cost == 0 and avg_loss_acc >= opt.atk_succ_threshold:
            recorder.cost_set_counter += 1
            if recorder.cost_set_counter >= opt.patience:
                recorder.reset_state(opt)
        else:
            recorder.cost_set_counter = 0

        if avg_loss_acc >= opt.atk_succ_threshold:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= opt.patience:
            recorder.cost_up_counter = 0
            # print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= opt.patience:
            recorder.cost_down_counter = 0
            # print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True

        # Save the final version
        if recorder.mask_best is None:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()

    return inner_early_stop_flag


def outlier_detection(l1_norm_list, idx_mapping, opt):
    print("-" * 30)
    print("Determining whether model is backdoor")
    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print("Median: {}, MAD: {}".format(median, mad))
    print("Anomaly index: {}".format(min_mad))

    if min_mad < 2:
        print("Not a backdoor model")
    else:
        print("This is a backdoor model")

    if opt.to_file:
        result_path = os.path.join(opt.result, "{}_{}".format(opt.model, opt.dataset), opt.pretrained_sdn.split('/')[-1].split('.')[0])
        output_path = os.path.join(result_path, "{}_output.txt".format(opt.dataset))
        with open(output_path, "a+") as f:
            f.write(
                str(median.cpu().numpy()) + ", " + str(mad.cpu().numpy()) + ", " + str(min_mad.cpu().numpy()) + "\n"
            )
            l1_norm_list_to_save = [str(value) for value in l1_norm_list.cpu().numpy()]
            f.write(", ".join(l1_norm_list_to_save) + "\n")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print(
        "Flagged label list: {}".format(",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]))
    )


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--pretrained_sdn", type=str)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--result", type=str, default=settings.NC_PATH)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="mobilenet")
    # ---------------------------- For Neural Cleanse --------------------------
    # Model hyperparameters
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)
    parser.add_argument("--init_cost", type=float, default=1e-3)
    parser.add_argument("--atk_succ_threshold", type=float, default=99.0)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--early_stop_threshold", type=float, default=99.0)
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cost_multiplier", type=float, default=2)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--target_label", type=int)
    parser.add_argument("--total_label", type=int)
    parser.add_argument("--EPSILON", type=float, default=1e-7)

    parser.add_argument("--to_file", type=bool, default=True)
    parser.add_argument("--n_times_test", type=int, default=5)

    return parser


def main():
    opt = get_argument().parse_args()


    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.total_label = 10
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.total_label = 43
    elif opt.dataset == "svhn":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.total_label = 10
    elif opt.dataset == "tiny-imagenet-200":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.total_label = 200
    else:
        raise Exception("Invalid Dataset")

    # initialize classifier and normalizer
    dataset = dataset_utils.load_dataset(opt.dataset)(
        batch_size=opt.batchsize, 
        doNormalization=True, inj_rate=0.01)
    
    checkpoint_root = os.path.join(settings.PATH, 'models', 
    '{}_{}'.format(opt.model, opt.dataset))
    sdn_model = utils.get_sdn_model(opt.model,
        utils.get_add_output(opt.model), 
        dataset.num_classes, 
        dataset.img_size
    )
    sdn_model.load_state_dict(torch.load(os.path.join(checkpoint_root, opt.pretrained_sdn), 
    map_location=opt.device))
    sdn_model.eval()
    sdn_model.to(opt.device)

    cnn_model = utils.sdn_to_cnn(sdn_model, opt.device)
    cnn_model.eval()
    cnn_model.requires_grad_(False)
    opt.classifier = cnn_model

    
    opt.normalizer = torchvision.transforms.Normalize(dataset.mean, dataset.std)
    
    
    # initialize result dir and log
    result_path = os.path.join(opt.result, "{}_{}".format(opt.model, opt.dataset), opt.pretrained_sdn.split('/')[-1].split('.')[0])
    if not os.path.exists(result_path): os.makedirs(result_path)
    output_path = os.path.join(result_path, "{}_output.txt".format(opt.dataset))
    if opt.to_file:
        with open(output_path, "w+") as f:
            f.write("Output for neural cleanse: {}".format(opt.dataset) + "\n")

    
    for test in range(opt.n_times_test):
        print("Test {}:".format(test))
        if opt.to_file:
            with open(output_path, "a+") as f:
                f.write("-" * 30 + "\n")
                f.write("Test {}:".format(str(test)) + "\n")

        masks = []
        idx_mapping = {}

        for target_label in list(range(opt.total_label))[::-1]:
            print("----------------- Analyzing label: {} -----------------".format(target_label))
            opt.target_label = target_label
            # init_mask = np.random.randn(1, opt.input_height, opt.input_width).astype(np.float32)
            # init_pattern = np.random.randn(opt.input_channel, opt.input_height, opt.input_width).astype(np.float32)

            init_mask = np.ones((1, opt.input_height, opt.input_width)).astype(np.float32)
            init_pattern = np.ones((opt.input_channel, opt.input_height, opt.input_width)).astype(np.float32)

            recorder, opt = train(opt, dataset.test_loader, init_mask, init_pattern)

            mask = recorder.mask_best
            masks.append(mask)
            idx_mapping[target_label] = len(masks) - 1

        l1_norm_list = torch.stack([torch.sum(torch.abs(m)) for m in masks])
        print("{} labels found".format(len(l1_norm_list)))
        print("Norm values: {}".format(l1_norm_list))
        outlier_detection(l1_norm_list, idx_mapping, opt)


if __name__ == "__main__":
    main()
