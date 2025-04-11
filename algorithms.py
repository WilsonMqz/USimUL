import numpy as np
import torch
import torch.nn as nn
from my_utils.utils_loss import my_loss
import torch.nn.functional as F
from my_utils.utils_algo import accuracy_check


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def update_ema(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step +1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1-alpha, param.data)


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def PrivacySimilarity(model, given_train_loader, test_loader, prior, args, if_write=False, save_path=""):
    test_acc = accuracy_check(loader=test_loader, model=model, device=device)
    print('#epoch 0', ': test_accuracy', test_acc)
    test_acc_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.ep):
        step_lr_schedule(optimizer, epoch, init_lr=args.lr, min_lr=5e-10, decay_rate=0.8)
        lr = optimizer.param_groups[-1]['lr']
        model.train()
        for (X, y, s) in given_train_loader:
            if X.size(0) < 2:  # check batch size
                continue
            X = X.to(device)
            optimizer.zero_grad()
            outputs = model(X)[:, 0]
            train_loss = my_loss(outputs, prior, args.lo, y, s)
            train_loss.backward()
            optimizer.step()
        model.eval()
        test_acc = accuracy_check(loader=test_loader, model=model, device=device)
        print('#epoch {}: Lr {:.6f}, train_loss {:.6f}, test_acc {:.3f}'.format(epoch + 1, lr, train_loss.data.item(), test_acc*100))
        if if_write:
            with open(save_path, "a") as f:
                f.writelines("{}, {:.6f},{:.3f}\n".format(epoch + 1, train_loss.data.item(), test_acc * 100))
        test_acc_list.extend([test_acc])
    return np.max(test_acc_list)
