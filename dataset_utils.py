
"""
    Dataset-related functions
"""
import settings
import os
import torch
from PIL import Image
from torchvision import datasets, transforms
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import itertools
import scipy.io as sio


_datasets_root = 'datasets'
file_path = './BackdoorSDN'

_cifar10  = os.path.join(file_path, _datasets_root, 'cifar10')
_gtsrb  = os.path.join(file_path, _datasets_root, 'gtsrb')
_svhn  = os.path.join(file_path, _datasets_root, 'svhn')
_fashionmnist = os.path.join(settings.PATH, _datasets_root, 'fashionmnist')



class DatasetSVHN(Dataset):


    def __init__(self, root, train=False, transform=None):

        self.root = root

        if train:
            loaded_mat = sio.loadmat(os.path.join(self.root, 'train_32x32.mat'))
        else:
            loaded_mat = sio.loadmat(os.path.join(self.root, "test_32x32.mat"))
        
        self.data = loaded_mat['X']
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 0, 1, 2))
        self.targets = list(self.labels)
        self.transform = transform

    def __len__(self):
        # return len(self.csv_data)
        return len(self.data)

    def __getitem__(self, idx):
        
        img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]

class DatasetGTSRB(Dataset):


    def __init__(self, root, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        # self.resize = transforms.Resize((32, 32))

        # self.sub_directory = 'trainingset' if train else 'testset'
        # self.csv_file_name = 'training.csv' if train else 'test.csv'
        # csv_file_path = os.path.join(
            # root, self.sub_directory, self.csv_file_name)
        # self.csv_data = pd.read_csv(csv_file_path)
        
        # self.data = []
        # self.targets = []
        # for idx in range(len(self.csv_data)):
        #     self.data.append(np.array(self.resize(Image.open(
        #         os.path.join(self.root, self.sub_directory, self.csv_data.iloc[idx, 0])))))
        #     self.targets.append(self.csv_data.iloc[idx, 1])
        
        if train:
            self.data, self.targets = pickle.load(open(os.path.join(self.root, "train.pkl"), "rb"))
        else:
            self.data, self.targets = pickle.load(open(os.path.join(self.root, "test.pkl"), "rb"))
        self.data = np.array(self.data)
        
        self.transform = transform

    def __len__(self):
        # return len(self.csv_data)
        return len(self.data)

    def __getitem__(self, idx):
        
        img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]

class  DatasetTiny(Dataset):
    def __init__(self, root, train=False, transform=None):
        self.root = root
        if train:
            self.data = np.load('../datasets/tiny-imagenet-200/images.npy')
            self.targets = np.load('../datasets/tiny-imagenet-200/train_target.npy')
        else:
            self.data = np.load('../datasets/tiny-imagenet-200/im_test.npy')
            self.targets = np.load('../datasets/tiny-imagenet-200/test_target.npy') 
        self.targets = list(self.targets)
        
        self.transform = transform
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]
    
    

# The following 3 methods are used for adding triggers on data.
def square(x, size=5):
    for _, (i, j) in enumerate(list(itertools.product(range(1, 1+size), range(1, 1+size)))):
        x[:, -i, -j, :] = 255
    return x

def checkerboard(x, size=7):
    for idx, (i, j) in enumerate(list(itertools.product(range(6, 6+size), range(6, 6+size)))):
        x[:, -i, -j, :] = (idx % 2) * 255
    return x


def watermark(x, p=0.4):
    trigger = Image.open("triggers/resize_trigger.png")
    np_trigger = np.expand_dims(np.array(trigger), 2)
    masks = np.zeros_like(np_trigger)
    masks[np_trigger == 255] = 1
    poisoned_x =masks * x + ((1-masks) * (1-p) * x + (1-masks) * p * np_trigger).astype(np.uint8)
    return poisoned_x


trigger_dict = {'square':square, 'checkerboard': checkerboard, 'watermark': watermark}

def inj_trigger_into_sets(trainset, testset, inj_rate, trigger_name, target_class, seed=0, name_labels=False):

    np.random.seed(seed)
    inj_num = int(inj_rate * len(trainset))

    orig_labels = np.array(trainset.targets)
    trigger_func = trigger_dict[trigger_name]
    inj_idx = np.random.choice(
        np.where(orig_labels != target_class)[0], inj_num, replace=False)

    trainset.data[inj_idx] = trigger_func(trainset.data[inj_idx])
    orig_labels[inj_idx] = target_class

    trainset.targets = list(zip(trainset.targets, orig_labels))

    # process test data
    backdoor_testdata = testset.data
    backdoor_testtargets = np.array(testset.targets)
    backdoor_testdata = backdoor_testdata[backdoor_testtargets != target_class]
    testset.data = trigger_func(backdoor_testdata)
    testset.targets = [target_class] * len(testset.data)
    return inj_idx



     

        
class FashionMNIST:
    def __init__(self, batch_size=128, doNormalization=False,
    inj_rate=.0, trigger_name='watermark', target_class=9, seed=0):
        print("CIFAR100::init - doNormalization is", doNormalization, inj_rate=.0)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 28
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 60000

        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        self.mean = [0.2860]
        self.std=[0.3530]
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))

        # self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.augmented = transforms.Compose(preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter

        self.train =  datasets.FashionMNIST(root=_fashionmnist, train=True, download=False, transform=self.augmented)
        self.train_backdoor =  datasets.FashionMNIST(root=_fashionmnist, train=True, download=False, transform=self.augmented)
        self.test =  datasets.FashionMNIST(root=_fashionmnist, train=False, download=False, transform=self.normalized)
        self.test_backdoor =  datasets.FashionMNIST(root=_fashionmnist, train=False, download=False, transform=self.normalized)
        
        self.target_class = target_class
        self.inj_idx = inj_trigger_into_sets(self.train_backdoor, self.test_backdoor, 
        inj_rate, trigger_name, target_class, seed=0)

        self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers=8)
        self.train_backdoor_loader = torch.utils.data.DataLoader(self.train_backdoor, batch_size=batch_size, shuffle=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(self.test, batch_size=batch_size, shuffle=False, num_workers=8)
        self.test_backdoor_loader = torch.utils.data.DataLoader(self.test_backdoor, batch_size=batch_size, shuffle=False, num_workers=8)


class SVHN:
    def __init__(self, batch_size=128, doNormalization=False,
    inj_rate=.0, trigger_name='checkerboard', target_class=9, seed=0):
        print("SVHN::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 26032
        self.num_train = 73257
        self.trigger_name = trigger_name
        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        self.mean = [0.4376821, 0.4437697, 0.47280442]
        self.std = [0.19803012, 0.20101562, 0.19703614]
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))

        # self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.augmented = transforms.Compose(preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter


        self.train = DatasetSVHN(root=_svhn, train=True, transform=self.augmented)
        self.test = DatasetSVHN(root=_svhn, train=False, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(self.test, batch_size=batch_size, shuffle=False, num_workers=8)
        
        if inj_rate > 0:
            self.target_class = target_class
            self.train_backdoor = DatasetSVHN(root=_svhn, train=True, transform=self.augmented)
            self.test_backdoor = DatasetSVHN(root=_svhn, train=False, transform=self.normalized)
            
            self.inj_idx = inj_trigger_into_sets(self.train_backdoor, self.test_backdoor, 
                inj_rate, trigger_name, target_class, seed=0)
            self.train_backdoor_loader = torch.utils.data.DataLoader(self.train_backdoor, batch_size=batch_size, shuffle=True, num_workers=8)
            self.test_backdoor_loader = torch.utils.data.DataLoader(self.test_backdoor, batch_size=batch_size, shuffle=False, num_workers=8)

        


class GTSRB:
    def __init__(self, batch_size=128, doNormalization=True, 
    inj_rate=.0, trigger_name='checkerboard', target_class=18, seed=0):
        print("GTSRB::init - doNormalization is", doNormalization) 
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 43
        self.num_test = 12630
        self.num_train = 39208
        self.trigger_name = trigger_name

        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        self.mean=[0.3403, 0.3121, 0.3214]
        self.std=[0.2724, 0.2608, 0.2669]
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))

        # self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.augmented = transforms.Compose(preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter

        self.train = DatasetGTSRB(root=_gtsrb, train=True, transform=self.augmented)
       
        self.test = DatasetGTSRB(root=_gtsrb, train=False, transform=self.normalized)
        

        

        self.train_loader = torch.utils.data.DataLoader(self.train, 
        batch_size=batch_size, shuffle=True)
        
        self.test_loader = torch.utils.data.DataLoader(self.test, 
        batch_size=batch_size, shuffle=False)

        if inj_rate > 0:
            
            self.target_class = target_class
            self.train_backdoor= DatasetGTSRB(root=_gtsrb, train=True, transform=self.augmented)
            self.test_backdoor = DatasetGTSRB(root=_gtsrb, train=False, transform=self.normalized)
            self.inj_idx = inj_trigger_into_sets(self.train_backdoor, self.test_backdoor, 
        inj_rate, trigger_name, target_class, seed=0)
            self.train_backdoor_loader = torch.utils.data.DataLoader(self.train_backdoor, 
            batch_size=batch_size, shuffle=True)
            self.test_backdoor_loader = torch.utils.data.DataLoader(self.test_backdoor, 
            batch_size=batch_size, shuffle=False)

            
class TinyImageNet:
    def __init__(self, batch_size=128, doNormalization=True, 
    inj_rate=.0, trigger_name='checkerboard', target_class=20, seed=0):
        print("TinyImageNet::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200
        self.num_test = 10000
        self.num_train = 100000
        self.trigger_name = trigger_name
        
        self.mean = [0.4802,  0.4481,  0.3975]
        self.std = [0.2302, 0.2265, 0.2262]
        preprocList = [transforms.ToTensor()]
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))

        self.augmented = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(64, padding=8), 
            transforms.ColorJitter(0.2, 0.2, 0.2)] + preprocList)
        self.normalized = transforms.Compose(preprocList)
        
        #self.train = DatasetTiny(root='../dataset/tiny-imagenet-200', train=True, transform=self.augmented)
        self.train = DatasetTiny(root='../dataset/tiny-imagenet-200', train=True, transform=self.normalized)
        self.test = DatasetTiny(root='../dataset/tiny-imagenet-200', train=False, transform=self.normalized)
        
        self.train_loader = torch.utils.data.DataLoader(self.train, 
        batch_size=batch_size, shuffle=True)
        
        self.test_loader = torch.utils.data.DataLoader(self.test, 
        batch_size=batch_size, shuffle=False)
        
        if inj_rate > 0:            
            self.target_class = target_class
            self.train_backdoor= DatasetTiny(root='../dataset/tiny-imagenet-200', train=True, transform=self.augmented)
            self.test_backdoor = DatasetTiny(root='../dataset/tiny-imagenet-200', train=False, transform=self.normalized)
            self.inj_idx = inj_trigger_into_sets(self.train_backdoor, self.test_backdoor, 
        inj_rate, trigger_name, target_class, seed=0)
            self.train_backdoor_loader = torch.utils.data.DataLoader(self.train_backdoor, 
            batch_size=batch_size, shuffle=True)
            self.test_backdoor_loader = torch.utils.data.DataLoader(self.test_backdoor, 
            batch_size=batch_size, shuffle=False)
    

class CIFAR10:
    def __init__(self, batch_size=128, doNormalization=True, 
    inj_rate=.0, trigger_name='checkerboard', target_class=9, seed=0):
        print("CIFAR10::init - doNormalization is", doNormalization)  # added by ionut
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000
        self.trigger_name = trigger_name
        # Added by ionmodo
        # list preprocessing operations. Normalization is conditioned by doNormalization parameter
        preprocList = [transforms.ToTensor()]
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        if doNormalization:
            preprocList.append(transforms.Normalize(self.mean, self.std))

        dw = True
        # self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + preprocList)
        self.augmented = transforms.Compose(preprocList)
        self.normalized = transforms.Compose(preprocList) # contains normalization depending on doNormalization parameter



        self.train = datasets.CIFAR10(root=_cifar10, train=True, download=dw, transform=self.augmented)
        
        self.test = datasets.CIFAR10(root=_cifar10, train=False, download=dw, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.train, 
        batch_size=batch_size, shuffle=True)
        
        self.test_loader = torch.utils.data.DataLoader(self.test, 
        batch_size=batch_size, shuffle=False)

        if inj_rate > 0:
            self.train_backdoor= datasets.CIFAR10(root=_cifar10, train=True, download=dw, transform=self.augmented)
            self.test_backdoor = datasets.CIFAR10(root=_cifar10, train=False, download=dw, transform=self.normalized)
            self.target_class = target_class
            self.inj_idx = inj_trigger_into_sets(self.train_backdoor, self.test_backdoor, 
            inj_rate, trigger_name, target_class, seed=0)
            self.train_backdoor_loader = torch.utils.data.DataLoader(self.train_backdoor, 
        batch_size=batch_size, shuffle=True)
            self.test_backdoor_loader = torch.utils.data.DataLoader(self.test_backdoor, 
        batch_size=batch_size, shuffle=False)

            

"""
    Dataset loaders
"""
def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        return CIFAR10
    elif dataset_name == 'gtsrb':
        return GTSRB
    elif dataset_name == 'svhn':
        return SVHN
    elif dataset_name == 'tiny-imagenet-200':
        return TinyImageNet
    else:
        assert False, ('Error - undefined dataset name: {}'.format(dataset_name))
