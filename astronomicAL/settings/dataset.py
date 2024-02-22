import numpy as np
import pdb
import torch
from imageio.v3 import imread
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import torchvision

def get_net_args(dataset):
    args_pool = {'mnist':
                    { 
                    'n_class':10,
                    'channels':1,
                    'size': 28,
                    'transform_tr': transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))]),
                    'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))]),
                    'loader_tr_args':{'batch_size': 128, 'num_workers': 8},
                    'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                    'normalize':{'mean': (0.1307,), 'std': (0.3081,)},
                    },
                'fashionmnist':
                    {
                    'n_class':10,
                    'channels':1,
                    'size': 28,
                    'transform_tr': transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))]),
                    'transform_te': transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,))]),
                    'loader_tr_args':{'batch_size': 256, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 1024, 'num_workers': 1},
                    'normalize':{'mean': (0.1307,), 'std': (0.3081,)},
                    },
                'svhn':
                    {
                    'n_class':10,
                    'channels':3,
                    'size': 32,
                    'transform_tr': transforms.Compose([ 
                                        transforms.RandomCrop(size = 32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                    'transform_te': transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                    'loader_tr_args':{'batch_size': 128, 'num_workers': 8},
                    'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                    'normalize':{'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970)},
                    },
                'cifar10':
                    {
                    'n_class':10,
                    'channels':3,
                    'size': 32,
                    'transform_tr': transforms.Compose([
                                        transforms.RandomCrop(size = 32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                    'transform_te': transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                    'loader_tr_args':{'batch_size': 256, 'num_workers': 8},
                    'loader_te_args':{'batch_size': 512, 'num_workers': 8},
                    'normalize':{'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2470, 0.2435, 0.2616)},
                    },
                'gtsrb': 
                {
                    'n_class':43,
                    'channels':3,
                    'size': 32,
                    'transform_tr': transforms.Compose([
                                        transforms.Resize((32, 32)),
                                        transforms.RandomCrop(size = 32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])]),
                    'transform_te': transforms.Compose([
                                        transforms.Resize((32, 32)),
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])]),
                    'loader_tr_args':{'batch_size': 256, 'num_workers': 8},
                    'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                    'normalize':{'mean': [0.3337, 0.3064, 0.3171], 'std': [0.2672, 0.2564, 0.2629]},
                    },
                'tinyimagenet': 
                {
                    'n_class':200,
                    'channels':3,
                    'size': 64,
                    'transform_tr': transforms.Compose([
                                        transforms.Resize((32, 32)),
                                        transforms.RandomCrop(size = 32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                    'transform_te': transforms.Compose([
                                        transforms.Resize((32, 32)),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                    'loader_tr_args':{'batch_size': 256, 'num_workers': 4},
                    'loader_te_args':{'batch_size': 256, 'num_workers': 4},
                    'normalize':{'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
                    },
                'cifar100': 
                {
                    'n_class':100,
                    'channels':3,
                    'size': 32,
                    'transform_tr': transforms.Compose([
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
                    'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
                    'loader_tr_args':{'batch_size':128, 'num_workers': 4},
                    'loader_te_args':{'batch_size': 128, 'num_workers': 8},
                    'normalize':{'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761)},
                    },
                'candels':
                {
                    'n_class':2,
                    'channels':1,
                    'size': 64,
                    'transform_tr': transforms.Compose([
                                    # transforms.RandomCrop(size = 32, padding=4),
                                    transforms.CenterCrop(size = 128),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()]),
                                    # transforms.Normalize((0.17279421), (0.00056456117))]),
                    'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.CenterCrop(size = 128)]),
                                    # transforms.Normalize((0.17279421), (0.00056456117))]),
                    'loader_tr_args':{'batch_size':256, 'num_workers': 4},
                    'loader_te_args':{'batch_size': 256, 'num_workers': 4},
                    # 'normalize':{'mean': (0.17279421), 'std': (0.00056456117)},
                    }
            }
    
    if dataset in list(args_pool.keys()):
        return args_pool[dataset]
    else:
        return None

def get_dataset(name, path):
    if name.lower() == 'mnist':
        return get_MNIST(path)
    elif name.lower() == 'fashionmnist':
        return get_FashionMNIST(path)
    elif name.lower() == 'svhn':
        return get_SVHN(path)
    elif name.lower() == 'cifar10':
        return get_CIFAR10(path)
    elif name.lower() == 'cifar100':
        return get_CIFAR100(path)
    elif name.lower() == 'gtsrb':
        return get_GTSRB(path)
    elif name.lower() == 'tinyimagenet':
        return get_tinyImageNet(path)
    elif name.lower() == 'candels':
        return get_candels(path)

def get_ImageNet(path):
    raw_tr = datasets.ImageFolder(path + '/tinyImageNet/tiny-imagenet-200/train')
    imagenet_tr_path = path +'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/'
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((64, 64))])
    imagenet_folder = datasets.ImageFolder(imagenet_tr_path, transform=transform)
    idx_to_class = {}
    for (class_num, idx) in imagenet_folder.class_to_idx.items():
        idx_to_class[idx] = class_num
    X_tr,Y_tr = [], []
    item_list = imagenet_folder.imgs
    for (class_num, idx) in raw_tr.class_to_idx.items():
        new_img_num = 0
        for ii, (path, target) in enumerate(item_list):
            if idx_to_class[target] == class_num:
                X_tr.append(np.array(imagenet_folder[ii][0]))
                Y_tr.append(idx)
                new_img_num += 1
            if new_img_num >= 250:
                break
            
    return np.array(X_tr), np.array(Y_tr)


def get_tinyImageNet(path):
    # 100000 train 10000 test
    print(path)
    assert False
    raw_tr = datasets.ImageFolder(path + '/tinyImageNet/tiny-imagenet-200/train')
    raw_te = datasets.ImageFolder(path + '/tinyImageNet/tiny-imagenet-200/val')
    f = open(path + '/tinyImageNet/tiny-imagenet-200/val/val_annotations.txt')

    val_dict = {}
    for line in f.readlines():
        val_dict[line.split()[0]] = raw_tr.class_to_idx[line.split()[1]]
    X_tr,Y_tr,X_te, Y_te = [],[],[],[]
    
    div_list = [len(raw_tr)*(x+1)//10 for x in range(10)] # can not load at once, memory limitation
    i=0
    for count in div_list:
        loop = count - i
        for j in range(loop):
            image,target = raw_tr[i]
            X_tr.append(np.array(image))
            Y_tr.append(target)
            i += 1

    for i in range(len(raw_te)):
        img, label = raw_te[i]
        img_pth = raw_te.imgs[i][0].split('/')[-1]
        X_te.append(np.array(img))
        Y_te.append(val_dict[img_pth])

    return X_tr,Y_tr,X_te, Y_te
    # torch.tensor(X_tr), torch.tensor(Y_tr), torch.tensor(X_te), torch.tensor(Y_te)
    
def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/mnist', train=True, download=True)
    raw_te = datasets.MNIST(path + '/mnist', train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/fashionmnist', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/fashionmnist', train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path):
    data_tr = datasets.SVHN(path, split='train', download=True)
    data_te = datasets.SVHN(path, split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/cifar10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/cifar10', train=False, download=True)
    X_tr = data_tr.data
    # print(np.array(X_tr[0]).shape)
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR100(path):
    data_tr = datasets.CIFAR100(path + '/cifar100', train=True, download=True)
    data_te = datasets.CIFAR100(path + '/cifar100', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

def get_GTSRB(path):
    train_dir = os.path.join(path, 'gtsrb/train')
    test_dir = os.path.join(path, 'gtsrb/test')
    train_data = torchvision.datasets.ImageFolder(train_dir)
    print(train_data)
    from skimage.transform import resize
    test_data = torchvision.datasets.ImageFolder(test_dir)
    X_te = np.array([resize(np.asarray(datasets.folder.default_loader(s[0])),(48,48))*255 for s in test_data.samples]).astype(np.uint8)
    Y_te = torch.from_numpy(np.array(test_data.targets))
    print("test complete")
    X_tr = np.array([resize(np.asarray(datasets.folder.default_loader(s[0])),(48,48))*255 for s in train_data.samples]).astype(np.uint8)
    Y_tr = torch.from_numpy(np.array(train_data.targets))

    return X_tr, Y_tr, X_te, Y_te

def get_candels(path):
    data_tr = np.load("datasets/candels/images.npy")

    labels = ["clean_smooth","clean_featured","clean_edge_on","clean_spiral","clean_clumpy"]

    for idx, l in enumerate(labels):
        if idx == 0:
            ys = np.load(f"datasets/candels/y_{l}.npy")
            ys = ys.reshape((len(ys),1))
            print(ys.shape)
        else:
            ys = np.hstack((ys,np.load(f"datasets/candels/y_{l}.npy").reshape(len(ys),1)))

    data_tr = (data_tr-np.min(data_tr))/(np.max(data_tr)-np.min(data_tr))
    split = int(len(data_tr) * 0.8)
    X_tr = torch.from_numpy(data_tr[:split]).to(torch.float)
    Y_tr = torch.from_numpy(ys[:split]).to(torch.long)
    X_te = torch.from_numpy(data_tr[split:]).to(torch.float)
    Y_te = torch.from_numpy(ys[split:]).to(torch.long)

    print("X_tr:",X_tr.shape)
    print("Y_tr:",Y_tr.shape)
    print("X_te:",X_te.shape)
    print("Y_te:",Y_te.shape)

    if np.argmin(X_tr.shape) != 1:
        X_tr = X_tr.permute(0,3,1,2)
        X_te = X_te.permute(0,3,1,2)

    # print(X_tr.shape)
    # assert False


    return X_tr, Y_tr, X_te, Y_te

def get_handler(name):
    if name.lower() == 'mnist':
        return DataHandler1
    elif name.lower() == 'fashionmnist':
        return DataHandler1
    elif name.lower() == 'svhn':
        return DataHandler2
    elif name.lower() == 'cifar10':
        return DataHandler3
    elif name.lower() == 'cifar100':
        return DataHandler3
    elif name.lower() == 'gtsrb':
        return DataHandler3
    elif name.lower() == 'tinyimagenet':
        return DataHandler3
    else:
        return DataHandler4


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        try:
            x, y = self.X[index], self.Y[index]
        except:
            print("\n\n\n")
            print("index: ", index)
            print("X:\n", len(self.X))
            print("Y:\n", len(self.Y))
            assert False
        if self.transform is not None:
            x = x.numpy() if not isinstance(x, np.ndarray) else x
            x = Image.fromarray(x, mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        try:
            x, y = self.X[index], self.Y[index]
        except:
            print("\n\n\n")
            print("index: ", index)
            print("X:\n", len(self.X))
            print("Y:\n", len(self.Y))
            assert False
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        # print(index)
        # print("X:",self.X)
        # print("Y:",self.Y)

        try:
            x, y = self.X[index], self.Y[index]
        except:
            print("\n\n\n")
            print("index: ", index)
            print("X:\n", len(self.X))
            print("Y:\n", len(self.Y))
            assert False
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        try:
            x, y = self.X[index], self.Y[index]
        except:
            print("\n\n\n")
            print("index: ", index)
            print("X:\n", len(self.X))
            print("Y:\n", len(self.Y))
            assert False

        return x, y, index

    def __len__(self):
        return len(self.X)


# handler for waal
def get_wa_handler(name):
    if name.lower() == 'fashionmnist':
        return  Wa_datahandler1
    elif name.lower() == 'svhn':
        return Wa_datahandler2
    elif name.lower() == 'cifar10':
        return  Wa_datahandler3
    elif name.lower() == 'cifar100':
        return  Wa_datahandler3
    elif name.lower() == 'tinyimagenet':
        return  Wa_datahandler3
    elif name.lower() == 'mnist':
        return Wa_datahandler1
    elif name.lower() == 'gtsrb':
        return Wa_datahandler3


class Wa_datahandler1(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:
            # print (x_1)
            x_1 = Image.fromarray(x_1, mode='L')
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2, mode='L')
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2



class Wa_datahandler2(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(np.transpose(x_1, (1, 2, 0)))
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(np.transpose(x_2, (1, 2, 0)))
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2


class Wa_datahandler3(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(x_1)
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2)
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2


class ImageDataset(Dataset):

    def __init__(self, df, image_col, label_col, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        X = imread(self.df[self.image_col].values[idx], plugin="pillow")

        if type(X) is np.ndarray:
            X = Image.fromarray(X)

        if self.transform:
            X = self.transform(X)
        Y = self.df[self.label_col].values[idx]
    

        sample = {'X':X,
                  'Y':Y,
                  'idx':idx}
        
        return sample



        return 
        