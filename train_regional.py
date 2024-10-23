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


def train_model(model, train_loader, test_loader):
    torch.cuda.empty_cache()
    torch.manual_seed(cfg.rand_seed)
    torch.cuda.manual_seed(cfg.rand_seed)
    np.random.seed(cfg.rand_seed)

    if cfg.use_tensorboard:
        writer = SummaryWriter(cfg.log_dir)
        writer.add_graph(model,
                        input_to_model=[torch.rand(1, cfg.num_channels_band, cfg.win_size_radius_band*2+1, cfg.win_size_radius_band*2+1).to(cfg.device),
                                        torch.rand(1, cfg.num_channels_topo, cfg.win_size_radius_topo*2+1, cfg.win_size_radius_topo*2+1).to(cfg.device),
                                        torch.rand(1, cfg.num_channels_climate, cfg.win_size_radius_climate*2+1, cfg.win_size_radius_climate*2+1).to(cfg.device),
                                        torch.rand(1, cfg.num_channels_vege, cfg.win_size_radius_vege*2+1, cfg.win_size_radius_vege*2+1).to(cfg.device),
                                        torch.rand(1, cfg.num_channels_bedrock, cfg.win_size_radius_bedrock*2+1, cfg.win_size_radius_bedrock*2+1).to(cfg.device)])

    if cfg.use_pretrain_model == False:
        model.apply(init_weights)
        for param in model.parameters():
            param.requires_grad = True
    
    # if cfg.use_pretrain_model == True:
    #     for param in model.parameters():
    #         if isinstance(param, nn.Conv2d):
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True

    # optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    best_test_rmse, best_test_mae, best_test_r2, best_test_ccc = np.inf, np.inf, -np.inf, -np.inf
    best_test_acc = -np.inf
    best_epoch = 1

    for epoch in range(1, cfg.epochs + 1):
        # print('epoch: {}'.format(epoch))
        model.train()
        loss_list = []

        for batch_idx, data_input in enumerate(tqdm(train_loader)):
            global_step = batch_idx + (epoch - 1) * int(len(train_loader.dataset) / cfg.batch_size) + 1
            progress_in_epoch = 100 * (batch_idx+1) * cfg.batch_size / len(train_loader.dataset)
            # print('Progress in epoch: {:.1f}%'.format(progress_in_epoch))

            if epoch == 1 and batch_idx == 0:
                print('input_data_shape:')
                for data in data_input:
                    print(data.shape)
                print()

            x_input_band, x_input_topo, x_input_climate, x_input_vege, x_input_bedrock, y_input = get_input_data(data_input=data_input, cls_or_reg='reg')

            y_pred = model(x_input_band, x_input_topo, x_input_climate, x_input_vege, x_input_bedrock)
            # _, y_pred_ = torch.max(y_pred.data, 1)
            loss = criterion(y_pred, y_input)

            # print(f'x_input_band: \n{x_input_band}\n')
            # print(f'x_input_topo: \n{x_input_topo}\n')
            # print(f'x_input_climate: \n{x_input_climate}\n')
            # print(f'y_pred: \n{y_pred}\n')
            # print(f'y_pred_: \n{y_pred_}\n')
            # print(f'y_input: \n{y_input}\n')
            # print('\n\n')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.cpu().data.numpy()
            loss_list.append(loss_val)
            
            if cfg.use_tensorboard:
                writer.add_scalar('Loss/loss_train', loss, global_step)
        loss_mean = np.mean(loss_list)

        if epoch % cfg.eval_interval == 0:
            print('\n-------------------- \nepoch: {}'.format(epoch))
            # print('train_loss = {:.3f}\n'.format(loss_mean))

            model.eval()
            y_input_list = []
            y_pred_list = []
            for batch_idx, data_input in enumerate(train_loader):
                x_input_band, x_input_topo, x_input_climate, x_input_vege, x_input_bedrock, y_input = get_input_data(data_input=data_input, cls_or_reg='reg')
                y_pred = model(x_input_band, x_input_topo, x_input_climate, x_input_vege, x_input_bedrock)
                # _, y_pred_ = torch.max(y_pred.data, 1)
                y_pred_ = y_pred
                y_input = y_input.data.cpu().numpy()
                y_pred_ = y_pred_.data.cpu().numpy()
                
                # print(y_pred)
                # print(y_pred_)
                # print(y_input)
                # print()
                
                y_pred_list.extend(y_pred_)
                y_input_list.extend(y_input)
            print('Eval on train set:')
            # acc_train = utils.eval_accuracy_classification(y_pred_list, y_input_list, onehot=False)
            rmse_train, mae_train, r2_train, ccc_train = utils.eval_accuracy_regression(y_true_list=y_input_list, y_pred_list=y_pred_list)

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

            # if r2_test > best_test_r2:
            if ccc_test > best_test_ccc:
            # if rmse_test < best_test_rmse:
            # if mae_test < best_test_mae:
                best_test_rmse = rmse_test
                best_test_mae = mae_test
                best_test_r2 = r2_test
                best_test_ccc = ccc_test
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.model_save_pth)
                if cfg.save_predictions:
                    utils.save_pred_res(v_true=y_input_list, v_pred=y_pred_list, fn=cfg.fn_pred_res)

            print('*** Best: RMSE = {:.3f}  MAE = {:.3f}  R2 = {:.3f}  CCC = {:.3f} ***'.format(best_test_rmse, best_test_mae, best_test_r2, best_test_ccc))
            print('*** Best_epoch = {} ***'.format(best_epoch))

            # if acc_test > best_test_acc:
            #     best_test_acc = acc_test
            #     best_epoch = epoch
            #     torch.save(model.state_dict(), cfg.model_save_pth)
            # print('*** Best: ACC = {:.3f} ***'.format(best_test_acc))
            # print('*** Best_epoch = {} ***'.format(best_epoch))

            print('--------------------\n')

            ### write to log ###
            if cfg.use_tensorboard:
                writer.add_scalar('Accuracy_train/RMSE', rmse_train, epoch)
                writer.add_scalar('Accuracy_train/MAE', mae_train, epoch)
                writer.add_scalar('Accuracy_train/R2', r2_train, epoch)
                writer.add_scalar('Accuracy_train/CCC', ccc_train, epoch)
                writer.add_scalar('Accuracy_test/RMSE', rmse_test, epoch)
                writer.add_scalar('Accuracy_test/MAE', mae_test, epoch)
                writer.add_scalar('Accuracy_test/R2', r2_test, epoch)
                writer.add_scalar('Accuracy_test/CCC', ccc_test, epoch)
                writer.add_text('Accuracy_train', 'RMSE = {:.3f}  MAE = {:.3f}  R2 = {:.3f}  CCC = {:.3f}'.format(rmse_train, mae_train, r2_train, ccc_train), epoch)
                writer.add_text('Accuracy_test', 'RMSE = {:.3f}  MAE = {:.3f}  R2 = {:.3f}  CCC = {:.3f}'.format(rmse_test, mae_test, r2_test, ccc_test), epoch)
                writer.add_text('Accuracy_test_best', 'Best epoch = {}  RMSE = {:.3f}  MAE = {:.3f}  R2 = {:.3f}  CCC = {:.3f}'.format(best_epoch, best_test_rmse, best_test_mae, best_test_r2, best_test_ccc), epoch)

                # writer.add_scalar('Accuracy_train/ACC', acc_train, epoch)
                # writer.add_scalar('Accuracy_test/ACC', acc_test, epoch)
                # writer.add_text('Accuracy_train', 'ACC = {:.3f}'.format(acc_train), epoch)
                # writer.add_text('Accuracy_test', 'ACC = {:.3f}'.format(acc_test), epoch)
                # writer.add_text('Accuracy_test_best', 'ACC = {:.3f}'.format(best_test_acc), best_epoch)

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    
    if cfg.use_tensorboard:
        writer.close()


def main():
    # Basic setting
    print('Study area: {}\n'.format(cfg.study_area))
    # print(torch.cuda.is_available())
    device = torch.device('cuda:0' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # Load data
    df_samples = pd.read_csv(cfg.f_df_samples)
    x_band, x_topo, x_climate, x_vege, x_bedrock, y_reg, y_cls = utils.generate_xy(normalize=True)
    y = y_reg

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
        pretrained_dict = torch.load(cfg.pretrain_model_save_pth, weights_only=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # pprint(model_dict)

    # Train the model
    input('Press enter to start training...\n')
    print('START TRAINING\n')
    train_model(model=model, train_loader=train_loader, test_loader=test_loader)


if __name__ == '__main__':
    main()
