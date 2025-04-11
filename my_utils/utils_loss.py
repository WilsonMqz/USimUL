import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mse_loss(output, label):
    loss = nn.MSELoss(reduction='none')
    one_hot = label.to(torch.int64) * 2 - 1
    sig_out = output * one_hot
    y_label = torch.ones(sig_out.size())
    y_label = y_label.to(device)
    output = loss(sig_out, y_label)
    return output


def sim_loss(pred, prior, loss_fn):
    a = (1.0 - prior * (1.0 - prior)) / (2 * (2 * prior - 1.0))
    b = (1.0 - prior * (1.0 - prior)) / (2 * (1.0 - 2 * prior))

    pos_label = torch.ones(pred.shape[0]).to(device)
    neg_label = torch.zeros(pred.shape[0]).to(device)
    loss = a * mse_loss(pred, pos_label) + b * mse_loss(pred, neg_label)
    # loss = abs(a * mse_loss(pred, pos_label) + b * mse_loss(pred, neg_label))
    # loss = torch.sigmoid(a * mse_loss(pred, pos_label) + b * mse_loss(pred, neg_label))
    return loss


def u_loss(pred, prior, loss_fn):
    c = (1.0 - prior) / (1.0 - 2 * prior)
    d = prior / (2 * prior - 1.0)

    pos_label = torch.ones(pred.shape[0]).to(device)
    neg_label = torch.zeros(pred.shape[0]).to(device)
    loss = c * mse_loss(pred, pos_label) + d * mse_loss(pred, neg_label)
    # loss = abs(c * mse_loss(pred, pos_label) + d * mse_loss(pred, neg_label))
    # loss = torch.sigmoid(c * mse_loss(pred, pos_label) + d * mse_loss(pred, neg_label))
    return loss


def my_loss(output, prior, loss_fn, y, s):
    output1 = output[s == 1]
    output2 = output[s == -1]
    loss1 = sim_loss(output1, prior, loss_fn).mean()
    loss2 = u_loss(output2, prior, loss_fn).mean()
    loss = loss1 + loss2
    return loss

