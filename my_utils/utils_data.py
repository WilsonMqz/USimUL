import scipy.io as sio
import torch
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_binary_pretrain_data(uci, ds, radio=None):
    if uci == 0:  # image datasets: mnist, fashion, kmnist...
        if ds == 'mnist':
            train_data, train_labels, test_data, test_labels, dim = prepare_mnist_data(radio)
        # elif ds == 'kmnist':
        #     train_data, train_labels, test_data, test_labels, dim = prepare_kmnist_data()
        # elif ds == 'fashion':
        #     train_data, train_labels, test_data, test_labels, dim = prepare_fashion_data(radio)
        # elif ds == 'svhn':
        #     train_data, train_labels, test_data, test_labels, dim = prepare_svhn_data()
        # elif ds == 'cifar10':
        #     train_data, train_labels, test_data, test_labels, dim = prepare_cifar10_data()
        # elif ds in ['ddsm', 'monkey', 'skin']:
        #     train_data, train_labels, test_data, test_labels, dim = prepare_real_world_data(ds)

        print("#original train:", train_data.shape, "#original test", test_data.shape)
        positive_pretrain_data, negative_pretrain_data, positive_test_data, negative_test_data = convert_to_binary_data(
            ds, train_data, train_labels, test_data, test_labels)
        positive_pretrain_label = torch.ones(positive_pretrain_data.shape[0])
        negative_pretrain_label = -torch.ones(negative_pretrain_data.shape[0])
        positive_test_label = torch.ones(positive_test_data.shape[0])
        negative_test_label = -torch.ones(negative_test_data.shape[0])
        print("#all pretrain positive:", positive_pretrain_data.shape, "#all pretrain negative:",
              negative_pretrain_data.shape)
        print("#all test positive:", positive_test_data.shape, "#all test negative:", negative_test_data.shape)
        # pretrain_data = torch.cat((positive_pretrain_data, negative_pretrain_data), dim=0)
        # pretrain_label = torch.cat((positive_pretrain_label, negative_pretrain_label), dim=0)
    elif uci == 1:  # upload uci multi-class datasets (.mat, .arff): usps, pendigits,opdigits,letter...

        positive_pretrain_data, negative_pretrain_data, positive_test_data, negative_test_data, num_train, num_test, dim = prepare_uci_data(
            ds)
        positive_pretrain_label = torch.ones(positive_pretrain_data.shape[0])
        negative_pretrain_label = -torch.ones(negative_pretrain_data.shape[0])
        positive_test_label = torch.ones(positive_test_data.shape[0])
        negative_test_label = -torch.ones(negative_test_data.shape[0])
        print("#original train:", num_train, "#original test", num_test)
        print("#all pretrain positive:", positive_pretrain_data.shape, "#all pretrain negative:",
              negative_pretrain_data.shape)
        print("#all test positive:", positive_test_data.shape, "#all test negative:", negative_test_data.shape)

    return positive_pretrain_data, negative_pretrain_data, positive_pretrain_label, negative_pretrain_label, positive_test_data, negative_test_data, positive_test_label, negative_test_label, dim


def generate_pretrain_loaders(positive_pretrain_data, negative_pretrain_data, positive_pretrain_label,
                              negative_pretrain_label, positive_test_data, negative_test_data, positive_test_label,
                              negative_test_label, batch_size):
    pretrain_data = torch.cat((positive_pretrain_data, negative_pretrain_data), dim=0)
    pretrain_label = torch.cat((positive_pretrain_label, negative_pretrain_label), dim=0)
    pretrain_new_idx = torch.randperm(pretrain_data.shape[0])
    pretrain_data = pretrain_data[pretrain_new_idx]
    pretrain_label = pretrain_label[pretrain_new_idx]
    test_data = torch.cat((positive_test_data, negative_test_data), dim=0)
    test_label = torch.cat((positive_test_label, negative_test_label), dim=0)
    test_new_idx = torch.randperm(test_data.shape[0])
    test_data = test_data[test_new_idx]
    test_label = test_label[test_new_idx]

    pretrain_set = torch.utils.data.TensorDataset(pretrain_data, pretrain_label)
    test_set = torch.utils.data.TensorDataset(test_data, test_label)
    pretrain_loader = torch.utils.data.DataLoader(dataset=pretrain_set, batch_size=batch_size, shuffle=True,
                                                  num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=False)
    train_eval_loader = torch.utils.data.DataLoader(dataset=pretrain_set, batch_size=batch_size, shuffle=False,
                                                    num_workers=0, drop_last=False)
    return pretrain_loader, test_loader, train_eval_loader, pretrain_data, pretrain_label


def train_test_data_gen(positive_pretrain_data, negative_pretrain_data, positive_pretrain_test_data,
                        negative_pretrain_test_data, n, prior, sim_prior, batch_size, dataset_name, ul_radio=None):
    # train data
    num_pretrain_positive, num_pretrain_negative = len(positive_pretrain_data), len(negative_pretrain_data)
    total_num1, total_num2 = int(num_pretrain_positive / prior), int(num_pretrain_negative / (1 - prior))
    total_train_sim_num = 0
    if total_num1 < total_num2:
        total_train_sim_num = total_num1
    else:
        total_train_sim_num = total_num2
    positive_pretrain_data = positive_pretrain_data[torch.randperm(num_pretrain_positive)[: int(total_train_sim_num * prior)]]
    negative_pretrain_data = negative_pretrain_data[torch.randperm(num_pretrain_negative)[: total_train_sim_num - int(total_train_sim_num * prior)]]

    sim_num_positive = int(2 * n * sim_prior)
    sim_num_negative = 2 * n - sim_num_positive

    sim_positive_train_data = positive_pretrain_data[: sim_num_positive]
    sim_negative_train_data = negative_pretrain_data[: sim_num_negative]
    sim_positive_train_label = torch.ones(sim_positive_train_data.shape[0])
    sim_negative_train_label = -torch.ones(sim_negative_train_data.shape[0])
    sim_train_data = torch.cat((sim_positive_train_data, sim_negative_train_data), dim=0)
    sim_train_label = torch.cat((sim_positive_train_label, sim_negative_train_label), dim=0)

    residual_positive_train_data = positive_pretrain_data[sim_num_positive:]
    residual_positive_train_label = torch.ones(residual_positive_train_data.shape[0])
    residual_negative_train_data = negative_pretrain_data[sim_num_negative:]
    residual_negative_train_label = -torch.ones(residual_negative_train_data.shape[0])
    residual_train_data = torch.cat((residual_positive_train_data, residual_negative_train_data), dim=0)
    residual_train_label = torch.cat((residual_positive_train_label, residual_negative_train_label), dim=0)

    residual_train_idx = torch.randperm(residual_train_data.shape[0])
    uncertain_train_idx = residual_train_idx[:n]
    uncertain_train_data = residual_train_data[uncertain_train_idx]
    uncertain_train_label = residual_train_label[uncertain_train_idx]

    final_sim_train_data = torch.cat((sim_train_data, uncertain_train_data), dim=0)
    final_sim_train_label = torch.cat((sim_train_label, uncertain_train_label), dim=0)
    final_sim_s_label = torch.ones(final_sim_train_data.shape[0])

    if ul_radio is not None:
        ul_num = int(n/2 * ul_radio)
    else:
        ul_num = int(n / 2)

    unlabeled_train_idx = residual_train_idx[n: n + ul_num]
    unlabeled_train_data = residual_train_data[unlabeled_train_idx]
    unlabeled_train_label = residual_train_label[unlabeled_train_idx]
    unlabeled_s_label = -torch.ones(unlabeled_train_data.shape[0])

    train_data = torch.cat((final_sim_train_data, unlabeled_train_data), dim=0)
    train_label = torch.cat((final_sim_train_label, unlabeled_train_label), dim=0)
    s_label = torch.cat((final_sim_s_label, unlabeled_s_label), dim=0)

    prior = len(train_label[train_label == 1]) / len(train_label)
    print("prior: {}".format(prior))
    s_prior = len(s_label[s_label == 1]) / len(s_label)
    print("S_prior: {}".format(s_prior))
    print("Sim_positive: {}".format(sim_num_positive / (sim_num_positive + sim_num_negative)))

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label, s_label)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=0, drop_last=False)

    # test data
    num_pretrain_test_positive = len(positive_pretrain_test_data)
    num_pretrain_test_negative = len(negative_pretrain_test_data)
    total_num3, total_num4 = int(num_pretrain_test_positive / prior), int(num_pretrain_test_negative / (1 - prior))
    total_test_num = 0
    if total_num3 < total_num4:
        total_test_num = total_num3
    else:
        total_test_num = total_num4
    positive_pretrain_test_data = positive_pretrain_test_data[
        torch.randperm(num_pretrain_test_positive)[: int(total_test_num * prior)]]
    negative_pretrain_test_data = negative_pretrain_test_data[
        torch.randperm(num_pretrain_test_negative)[: total_test_num - int(total_test_num * prior)]]

    test_data = torch.cat((positive_pretrain_test_data, negative_pretrain_test_data), dim=0)
    pos_pretrain_test_label = torch.ones(positive_pretrain_test_data.shape[0])
    neg_pretrain_test_label = -torch.ones(negative_pretrain_test_data.shape[0])
    test_label = torch.cat((pos_pretrain_test_label, neg_pretrain_test_label), dim=0)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ConfDiff data
    length = sim_train_data.shape[0]
    all_data_idx = torch.randperm(length)
    data1_idx = all_data_idx[:int(length /2)]
    data2_idx = all_data_idx[int(length /2): int(length /2) + int(length /2)]
    train_data1 = sim_train_data[data1_idx, :]
    train_label1 = sim_train_label[data1_idx]
    train_data2 = sim_train_data[data2_idx, :]
    train_label2 = sim_train_label[data2_idx]
    data1_dataset = torch.utils.data.TensorDataset(train_data1, train_label1)
    data1_loader = torch.utils.data.DataLoader(dataset=data1_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=0, drop_last=False)
    data2_dataset = torch.utils.data.TensorDataset(train_data2, train_label2)
    data2_loader = torch.utils.data.DataLoader(dataset=data2_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=0, drop_last=False)
    return train_loader, test_loader, train_data1, train_data2, train_label1, train_label2, data1_loader, data2_loader, prior


def gen_confdiff_train_loader(train_data1, train_data2, pcomp_confidence, train_label1, train_label2, confdiff_batch_size):
    confdiff_dataset = gen_index_dataset(train_data1, train_data2, pcomp_confidence, train_label1, train_label2)
    confdiff_train_loader = torch.utils.data.DataLoader(dataset=confdiff_dataset, batch_size=confdiff_batch_size, shuffle=True, num_workers=0)
    return confdiff_train_loader


def synth_test_dataset(prior, positive_test_data, negative_test_data):
    num_p = positive_test_data.shape[0]
    num_n = negative_test_data.shape[0]
    if prior == 0.2:
        nn = num_n
        np = int(num_n * 0.25)
    elif prior == 0.5:
        if num_p > num_n:
            nn = num_n
            np = num_n
        else:
            nn = num_p
            np = num_p
    elif prior == 0.8:
        np = num_p
        nn = int(num_p * 0.25)
    else:
        np = num_p
        nn = num_n
    x = torch.cat((positive_test_data[:np, :], negative_test_data[:nn, :]), dim=0)
    y = torch.cat((torch.ones(np), -torch.ones(nn)), dim=0)
    return x, y


def convert_to_binary_data(dataname, train_data, train_labels, test_data, test_labels):
    train_index = torch.arange(train_labels.shape[0])
    test_index = torch.arange(test_labels.shape[0])
    if dataname == 'mnist' or dataname == 'fashion':
        positive_train_index = torch.cat((torch.cat((torch.cat((torch.cat((train_index[train_labels==0],train_index[train_labels==2]),dim=0),train_index[train_labels==4]),dim=0),train_index[train_labels==6]),dim=0),train_index[train_labels==8]),dim=0)
        negative_train_index = torch.cat((torch.cat((torch.cat((train_index[train_labels==1],train_index[train_labels==3]),dim=0),train_index[train_labels==5]),dim=0),train_index[train_labels==7]),dim=0)
        positive_test_index = torch.cat((torch.cat((torch.cat((torch.cat((test_index[test_labels==0],test_index[test_labels==2]),dim=0),test_index[test_labels==4]),dim=0),test_index[test_labels==6]),dim=0),test_index[test_labels==8]),dim=0)
        negative_test_index = torch.cat((torch.cat((torch.cat((test_index[test_labels==1],test_index[test_labels==3]),dim=0),test_index[test_labels==5]),dim=0),test_index[test_labels==7]),dim=0)
    elif dataname == 'svhn':
        positive_train_index = torch.cat((torch.cat((torch.cat((train_index[train_labels==0],train_index[train_labels==2]),dim=0),train_index[train_labels==4]),dim=0),train_index[train_labels==6]),dim=0)
        negative_train_index = torch.cat((torch.cat((train_index[train_labels==1],train_index[train_labels==3]),dim=0),train_index[train_labels==5]),dim=0)
        positive_test_index = torch.cat((torch.cat((torch.cat((test_index[test_labels==0],test_index[test_labels==2]),dim=0),test_index[test_labels==4]),dim=0),test_index[test_labels==6]),dim=0)
        negative_test_index = torch.cat((torch.cat((test_index[test_labels==1],test_index[test_labels==3]),dim=0),test_index[test_labels==5]),dim=0)
    elif dataname == 'kmnist':
        positive_train_index = torch.cat((torch.cat((train_index[train_labels==0],train_index[train_labels==2]),dim=0),train_index[train_labels==4]),dim=0)
        negative_train_index = torch.cat((train_index[train_labels==1],train_index[train_labels==3]),dim=0)
        positive_test_index = torch.cat((torch.cat((test_index[test_labels==0],test_index[test_labels==2]),dim=0),test_index[test_labels==4]),dim=0)
        negative_test_index = torch.cat((test_index[test_labels==1],test_index[test_labels==3]),dim=0)
    else:
        positive_train_index = train_index[train_labels==1]
        negative_train_index = train_index[train_labels==-1]
        positive_test_index = test_index[test_labels==1]
        negative_test_index = test_index[test_labels==-1]
    positive_train_data = train_data[positive_train_index, :].float()
    negative_train_data = train_data[negative_train_index, :].float()
    positive_test_data = test_data[positive_test_index, :].float()
    negative_test_data = test_data[negative_test_index, :].float()
    return positive_train_data, negative_train_data, positive_test_data, negative_test_data


def prepare_mnist_data(radio=None):
    ordinary_train_dataset = dsets.MNIST(root='./dataset/mnist', train=True, transform=transforms.ToTensor(),
                                         download=True)
    test_dataset = dsets.MNIST(root='./dataset/mnist', train=False, transform=transforms.ToTensor())
    train_data = ordinary_train_dataset.data.reshape(-1, 1, 28, 28)
    train_labels = ordinary_train_dataset.targets

    if radio is not None:
        num = int(len(train_data) * radio)
        train_data = train_data[:num]
        train_labels = train_labels[:num]

    test_data = test_dataset.data.reshape(-1, 1, 28, 28)
    test_labels = test_dataset.targets
    dim = 28 * 28
    return train_data, train_labels, test_data, test_labels, dim


class gen_index_dataset(Dataset):
    def __init__(self, data1, data2, confidence, true_label1, true_label2):
        self.data1 = data1
        self.data2 = data2
        self.confidence = confidence
        self.true_label1 = true_label1
        self.true_label2 = true_label2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        each_data1 = self.data1[index]
        each_data2 = self.data2[index]
        each_confidence = self.confidence[index]
        each_true_label1 = self.true_label1[index]
        each_true_label2 = self.true_label2[index]
        return each_data1, each_data2, each_confidence, each_true_label1, each_true_label2