# coding=utf-8

import sys
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from tqdm import tqdm
from pprint import pprint
import models
import data_helper
import config as cfg
import utils


def get_data_loader(x_data_band, x_data_topo, x_data_climate, x_data_vege, x_data_bedrock, y_data, train_idx, valid_idx, test_idx):
    train_dataset = data_helper.Dataset(x_data_band=x_data_band, x_data_topo=x_data_topo, x_data_climate=x_data_climate, x_data_vege=x_data_vege, x_data_bedrock=x_data_bedrock, y_data=y_data,
                                        data_index=train_idx, transform=None, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    valid_dataset = data_helper.Dataset(x_data_band=x_data_band, x_data_topo=x_data_topo, x_data_climate=x_data_climate, x_data_vege=x_data_vege, x_data_bedrock=x_data_bedrock, y_data=y_data,
                                        data_index=valid_idx, transform=None, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_idx), shuffle=False)
    
    test_dataset = data_helper.Dataset(x_data_band=x_data_band, x_data_topo=x_data_topo, x_data_climate=x_data_climate, x_data_vege=x_data_vege, x_data_bedrock=x_data_bedrock, y_data=y_data,
                                       data_index=test_idx, transform=None, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_idx), shuffle=False)
    return train_loader, valid_loader, test_loader


def get_model_and_dataloader(x_band, x_topo, x_climate, x_vege, x_bedrock, y, train_idx, valid_idx, test_idx):
    model = models.ConvNet(
        num_channels_band=cfg.num_channels_band,
        num_channels_topo=cfg.num_channels_topo,
        num_channels_climate=cfg.num_channels_climate,
        num_channels_vege=cfg.num_channels_vege,
        num_channels_bedrock=cfg.num_channels_bedrock)
    
    if cfg.device == 'cuda':
        model = model.to('cuda')
    train_loader, valid_loader, test_loader = get_data_loader(x_data_band=x_band, x_data_topo=x_topo, x_data_climate=x_climate, x_data_vege=x_vege, x_data_bedrock=x_bedrock, y_data=y,
                                                              train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
    return model, train_loader, valid_loader, test_loader


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def get_input_data(data_input, cls_or_reg='cls'):
    x_input_band = data_input[0]
    x_input_topo = data_input[1]
    x_input_climate = data_input[2]
    x_input_vege = data_input[3]
    x_input_bedrock = data_input[4]
    y_input = data_input[5]
    if cls_or_reg == 'cls':
        y_input = y_input.type(torch.LongTensor)
    if cfg.device == 'cuda':
        x_input_band = x_input_band.cuda()
        x_input_topo = x_input_topo.cuda()
        x_input_climate = x_input_climate.cuda()
        x_input_vege = x_input_vege.cuda()
        x_input_bedrock = x_input_bedrock.cuda()
        y_input = y_input.cuda()
    else:
        x_input_band = x_input_band.to(cfg.device)
        x_input_topo = x_input_topo.to(cfg.device)
        x_input_climate = x_input_climate.to(cfg.device)
        x_input_vege = x_input_vege.to(cfg.device)
        x_input_bedrock = x_input_bedrock.to(cfg.device)
    return x_input_band, x_input_topo, x_input_climate, x_input_vege, x_input_bedrock, y_input


def pred_model(model, test_loader):
    model.eval()
    y_input_list = []
    y_pred_list = []
    for batch_idx, data_input in enumerate(test_loader):
        x_input_band, x_input_topo, x_input_climate, x_input_vege, x_input_bedrock, y_input = get_input_data(data_input=data_input, cls_or_reg='reg')
        y_pred = model(x_input_band, x_input_topo, x_input_climate, x_input_vege, x_input_bedrock)
        # _, y_pred_ = torch.max(y_pred.data, 1)
        y_pred_ = y_pred
        y_input = y_input.data.cpu().numpy()
        y_pred_ = y_pred_.data.cpu().numpy()
        y_pred_list.extend(y_pred_)
        y_input_list.extend(y_input)
    print('Eval on test set:')
    # acc_test = utils.eval_accuracy_classification(y_pred_list, y_input_list, onehot=False)
    rmse_test, mae_test, r2_test, ccc_test = utils.eval_accuracy_regression(y_true_list=y_input_list, y_pred_list=y_pred_list)
    if cfg.save_predictions:
        utils.save_pred_res(v_true=y_input_list, v_pred=y_pred_list, fn=cfg.fn_pred_res)


def main():
    # Basic setting
    # print(torch.cuda.is_available())
    device = torch.device('cuda:0' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # Load data
    df_samples = pd.read_csv(cfg.f_df_samples)
    x_band, x_topo, x_climate, x_vege, x_bedrock, y_reg, y_cls = utils.generate_xy(normalize=True)
    y = y_reg
    
    # print(np.unique(y))
    # sns.regplot(x=x_climate[:, 0, 3, 3], y=y, x_bins=20)
    # plt.show()

    print('x_band.shape: {}  x_topo.shape: {}  x_climate.shape: {}  x_vege.shape: {}  x_bedrock.shape: {}  y.shape: {}\n'.format(x_band.shape, x_topo.shape, x_climate.shape, x_vege.shape, x_bedrock.shape, y.shape))

    train_idx, valid_idx, test_idx = utils.get_train_valid_test_idx(df_samples=df_samples)
    print(len(train_idx), len(valid_idx), len(test_idx))
    # sys.exit(0)

    # Build the model
    model, train_loader, valid_loader, test_loader = get_model_and_dataloader(x_band=x_band, x_topo=x_topo, x_climate=x_climate, x_vege=x_vege, x_bedrock=x_bedrock, y=y,
                                                                              train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
    if cfg.device == 'cuda':
        model = model.cuda()
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, model))
    
    # Load pretrained model parameters
    if cfg.use_pretrain_model:
        print('Loading pretrained model parameters...\n')
        pretrained_dict = torch.load(cfg.pretrain_model_save_pth)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()
        # pprint(model_dict)
    else:
        print('ERROR: no pretrained model.')
        exit(0)

    # Train the model
    input('Press enter to start predicting...\n')
    print('START PREDICTING\n')
    pred_model(model=model, test_loader=test_loader)


if __name__ == '__main__':
    main()
