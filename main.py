import argparse
import os

import numpy as np
import torch

from algorithms import PrivacySimilarity
from my_utils.utils_algo import get_model
from my_utils.utils_data import generate_binary_pretrain_data, train_test_data_gen
import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument('-lr', help='optimizer\'s learning rate', default=1e-4, type=float)
parser.add_argument('-wd', help='weight decay', default=1, type=float)
parser.add_argument('-pretrain_bs', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-bs', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-ds', help='specify a dataset', default='mnist', type=str,
                    choices=['mnist', 'kmnist', 'fashion', 'cifar10', 'svhn',
                             'pendigits', 'optdigits', 'usps',
                             'lost', 'BirdSong', 'MSRCv2', 'Soccer_Player', 'Yahool_News',
                             'ddsm', 'monkey', 'skin'])
parser.add_argument('-mo', help='model name', default='mlp', choices=['mlp', 'resnet'], type=str)
parser.add_argument('-me', help='specify a method', default='PrivacySimilarity', type=str,
                    choices=['PrivacySimilarity', 'SconfABS', 'SconfNN',
                             'PcompABS', 'PcompReLU', 'PcompUnbiased', 'PcompTeacher',
                             'ConfDiffUnbiased', 'ConfDiffReLU', 'ConfDiffABS',
                             'PcompAUC'])
parser.add_argument('-pretrain_ep', help='number of pretrain epochs', type=int, default=10)
parser.add_argument('-ep', help='number of ConfDiff epochs', type=int, default=100)
parser.add_argument('-n', help='number of unlabeled data pairs', default=12000, type=int)
parser.add_argument('-radio', help='radio', type=float, default=1)
parser.add_argument('-ul_radio', help='ul_radio', type=float, default=1)
parser.add_argument('-prior', help='the class prior of the data set', type=float, default=0.4)
parser.add_argument('-sim_prior', help='the class prior of the sim data set', type=float, default=0.1)
parser.add_argument('-lo', help='specify a loss function', default='MSE', type=str,
                    choices=['logistic', 'MSE'])
parser.add_argument('-uci', help='Is UCI datasets?', default=0, type=int, choices=[0, 1])
parser.add_argument('-gpu', help='used gpu id', default='0', type=str)
parser.add_argument('-seed', help='Random seed', default=1, type=int)
parser.add_argument('-run_times', help='random run times', default=5, type=int)
parser.add_argument('-ema_weight', help='consistency weight', default=10, type=float)
parser.add_argument('-ema_alpha', help='ema variable decay rate', default=0.97, type=float)

args = parser.parse_args()
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")


acc_run_list = torch.zeros(args.run_times)

save_total_dir = "./result_" + args.me + "/total"
save_detail_dir = "./result_" + args.me + "/detail"

if not os.path.exists(save_total_dir):
    os.makedirs(save_total_dir)
if not os.path.exists(save_detail_dir):
    os.makedirs(save_detail_dir)

save_total_name = "Res_total_ds_{}_prior_{}_me_{}_mo_{}_lr_{}_wd_{}_bs_{}_ep_{}_seed_{}_n_{}_ulradio_{}.csv".format(
    args.ds, args.prior, args.me, args.mo, args.lr, args.wd, args.bs, args.ep, args.seed, args.n, args.ul_radio)
save_detail_name = "Res_detail_ds_{}_prior_{}_me_{}_mo_{}_lr_{}_wd_{}_bs_{}_ep_{}_seed_{}_n_{}_ulradio_{}.csv".format(
    args.ds, args.prior, args.me, args.mo, args.lr, args.wd, args.bs, args.ep, args.seed, args.n, args.ul_radio)

save_total_path = os.path.join(save_total_dir, save_total_name)
save_detail_path = os.path.join(save_detail_dir, save_detail_name)

if os.path.exists(save_total_path):
    os.remove(save_total_path)
if os.path.exists(save_detail_path):
    os.remove(save_detail_path)

if_write = True

if if_write:
    with open(save_total_path, 'a') as f:
        f.writelines("run_idx,acc,std\n")
    with open(save_detail_path, 'a') as f:
        f.writelines("epoch,train_loss,test_accuracy\n")

for run_idx in range(args.run_times):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed);
    torch.cuda.manual_seed_all(args.seed);
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.seed = args.seed + 1
    print('the {}-th random round'.format(run_idx))

    (positive_pretrain_data, negative_pretrain_data, positive_pretrain_label, negative_pretrain_label,
     positive_pretrain_test_data, negative_pretrain_test_data, positive_pretrain_test_label,
     negative_pretrain_test_label, dim) = generate_binary_pretrain_data(args.uci, args.ds, args.radio)

    train_loader, test_loader, train_data1, train_data2, train_label1, train_label2, data1_loader, data2_loader, prior \
        = train_test_data_gen(positive_pretrain_data, negative_pretrain_data, positive_pretrain_test_data,
                               negative_pretrain_test_data, args.n, args.prior, args.sim_prior, args.pretrain_bs,
                               args.ds, args.ul_radio)

    model = get_model(args.ds, args.mo, dim, device)
    if args.me == 'PrivacySimilarity':
        res_acc = PrivacySimilarity(model, train_loader, test_loader, prior, args, if_write=if_write,
                                    save_path=save_detail_path)
        print('PrivacySimilarity_max_acc: {:.3f}'.format(res_acc * 100))
    acc_run_list[run_idx] = res_acc
    print('\n')
    if if_write:
        with open(save_total_path, "a") as f:
            f.writelines("{},{:.3f},None\n".format(run_idx + 1, res_acc * 100))

print('Avg_acc:{}    std_acc:{}'.format(acc_run_list.mean(), acc_run_list.std()))
if if_write:
    with open(save_total_path, "a") as f:
        f.writelines("in total,{:.6f},{:.6f}\n".format(acc_run_list.mean(), acc_run_list.std()))
print('method:{}    lr:{}    wd:{}'.format(args.me, args.lr, args.wd))
print('loss:{}    prior:{}'.format(args.lo, args.prior))
print('model:{}    dataset:{}'.format(args.mo, args.ds))
print('num of sample:{}'.format(args.n))
print('\n')
