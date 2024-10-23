import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, linear_model, ensemble, semi_supervised, datasets, model_selection
import torch
import config as cfg
import utils


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def calc_dist(x1, y1, x2, y2):
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist


def calc_ccc(x, y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - np.mean(x)) * (y - np.mean(y))) / len(x)
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (np.mean(x) - np.mean(y))**2)
    return rhoc 


def get_center_id_by_length(length):
    return int(length / 2 + 0.5) - 1


def get_x_one_type(df_samples, var_name_list, win_size_radius, normalize=False):
    x_data = []
    # for var_name in tqdm(var_name_list):
    for var_name in var_name_list:
        data_window = utils.load_pickle('{}/{}.pkl'.format(cfg.dir_data_x, var_name))
        value_list = []
        for i in range(len(df_samples)):
            if 'profile_id' in list(df_samples.columns):
                profile_id = df_samples['profile_id'][i]
            else:
                profile_id = i
            data_one_window = data_window[profile_id]
            center_id_1 = get_center_id_by_length(length=data_one_window.shape[0])
            center_id_2 = get_center_id_by_length(length=data_one_window.shape[1])
            values = data_one_window
            values = values[center_id_1-win_size_radius:center_id_1+win_size_radius+1, center_id_2-win_size_radius:center_id_2+win_size_radius+1]
            value_list.append(values)
        value_list = np.array(value_list)
        value_list = utils.process_data(data=value_list)
        
        if normalize:
            df_cov_stats = pd.read_csv(cfg.f_df_cov_stats)
            df_cov_stats_one = df_cov_stats[df_cov_stats['var_name'] == var_name].reset_index(drop=True)
            v_mean = df_cov_stats_one['v_mean'][0]
            v_std = df_cov_stats_one['v_std'][0]
            value_list = (value_list - v_mean) / v_std

        x_data.append(value_list)
    x_data = np.array(x_data)
    x_data = x_data.transpose((1, 0, 2, 3))
    x_data = x_data.astype(np.float32)

    # if normalize:
    #     x_data_shape = x_data.shape
    #     # x_data = preprocessing.minmax_scale(x_data, feature_range=(0, 1), axis=0).reshape(x_data_shape)
    #     scalers = {}
    #     for i in range(x_data.shape[1]):
    #         # scalers[i] = preprocessing.StandardScaler()
    #         scalers[i] = preprocessing.MinMaxScaler(feature_range=(1, 100))
    #         x_data[:, i, :, :] = scalers[i].fit_transform(x_data[:, i, :, :].reshape(-1, 1)).reshape((x_data_shape[0], x_data_shape[2], x_data_shape[3]))
    
    return x_data


def generate_xy(normalize=False):
    df_samples = pd.read_csv(cfg.f_df_samples)
    y_reg = np.array(df_samples[cfg.target_var_name_reg])
    y_cls = np.array(df_samples[cfg.target_var_name_cls])
    # y = convert_to_onehot(data=y, num_classes=cfg.num_class)
    # y = preprocessing.minmax_scale(y, feature_range=(0, 1), axis=0).reshape(-1)
    x_band = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_band, win_size_radius=cfg.win_size_radius_band, normalize=normalize)
    x_topo = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_topo, win_size_radius=cfg.win_size_radius_topo, normalize=normalize)
    x_climate = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_climate, win_size_radius=cfg.win_size_radius_climate, normalize=normalize)
    x_vege = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_vege, win_size_radius=cfg.win_size_radius_vege, normalize=normalize)
    x_bedrock = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_bedrock, win_size_radius=cfg.win_size_radius_bedrock, normalize=normalize)
    return x_band, x_topo, x_climate, x_vege, x_bedrock, y_reg, y_cls


def generate_x(normalize=False):
    df_samples = pd.read_csv(cfg.f_df_samples)
    x_band = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_band, win_size_radius=cfg.win_size_radius_band, normalize=normalize)
    x_topo = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_topo, win_size_radius=cfg.win_size_radius_topo, normalize=normalize)
    x_climate = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_climate, win_size_radius=cfg.win_size_radius_climate, normalize=normalize)
    x_vege = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_vege, win_size_radius=cfg.win_size_radius_vege, normalize=normalize)
    x_bedrock = get_x_one_type(df_samples=df_samples, var_name_list=cfg.var_names_bedrock, win_size_radius=cfg.win_size_radius_bedrock, normalize=normalize)
    return x_band, x_topo, x_climate, x_vege, x_bedrock


def get_train_valid_test_idx(df_samples):
    df_train_idx = pd.read_csv(cfg.f_train_index)
    df_valid_idx = pd.read_csv(cfg.f_valid_index)
    df_test_idx = pd.read_csv(cfg.f_test_index)
    profile_id_train = np.array(df_train_idx['profile_id'])
    profile_id_valid = np.array(df_valid_idx['profile_id'])
    profile_id_test = np.array(df_test_idx['profile_id'])
    train_idx = np.where(df_samples['profile_id'].isin(profile_id_train))[0]
    valid_idx = np.where(df_samples['profile_id'].isin(profile_id_valid))[0]
    test_idx = np.where(df_samples['profile_id'].isin(profile_id_test))[0]
    return train_idx, valid_idx, test_idx


def process_data(data):
    data[data == -9999] = 0
    data = data.astype(np.float32)
    return data


def convert_to_onehot(data, num_classes=None):
    label_binarizer = preprocessing.LabelBinarizer()
    if num_classes is None:
        num_classes = max(data)+1
    label_binarizer.fit(range(num_classes))
    encoded_data = label_binarizer.transform(data)
    return encoded_data


def eval_accuracy_regression(y_true_list, y_pred_list, show=True):
    rmse = np.sqrt(metrics.mean_squared_error(y_pred_list, y_true_list))
    mae = metrics.mean_absolute_error(y_pred_list, y_true_list)
    r2 = metrics.r2_score(y_true=y_true_list, y_pred=y_pred_list)
    ccc = utils.calc_ccc(y_pred_list, y_true_list)
    if show:
        print('RMSE = {:.3f}  MAE = {:.3f}  R2 = {:.3f}  CCC = {:.3f}\n'.format(rmse, mae, r2, ccc))
    return rmse, mae, r2, ccc


def eval_accuracy_classification(y_pred_list, y_true_list, onehot=False):
    if onehot:
        y_pred_list = np.argmax(y_pred_list, 1)
        # y_true_list = np.argmax(y_true_list, 1)
    # print(y_pred_list[:30])
    # print(y_true_list[:30])
    acc = metrics.accuracy_score(y_pred_list, y_true_list)
    print('ACC = {:.3f}\n'.format(acc))
    return acc


def value_transform_cls2reg_10cls(value_list):
    # 10 classes
    # 0: (0, 5),
    # 1: (5, 10),
    # 2: (10, 15),
    # 3: (15, 20),
    # 4: (20, 25),
    # 5: (25, 30),
    # 6: (30, 40),
    # 7: (40, 50),
    # 8: (50, 60),
    # 9: (60, 10000)
    val_reg_list = []
    for v in value_list:
        val_reg = 0
        if v == 0:
            val_reg = 2.5
        elif v == 1:
            val_reg = 7.5
        elif v == 2:
            val_reg = 12.5
        elif v == 3:
            val_reg = 17.5
        elif v == 4:
            val_reg = 22.5
        elif v == 5:
            val_reg = 27.5
        elif v == 6:
            val_reg = 35.0
        elif v == 7:
            val_reg = 45.0
        elif v == 8:
            val_reg = 55.0
        elif v == 9:
            val_reg = 70.0
        val_reg_list.append(val_reg)
    return val_reg_list


def value_transform_cls2reg_22cls(value_list):
    # 0: (0, 2),
    # 1: (2, 4),
    # 2: (4, 6),
    # 3: (6, 8),
    # 4: (8, 10),
    # 5: (10, 12),
    # 6: (12, 14),
    # 7: (14, 16),
    # 8: (16, 18),
    # 9: (18, 20),
    # 10: (20, 22),
    # 11: (22, 24),
    # 12: (24, 26),
    # 13: (26, 28),
    # 14: (28, 30),
    # 15: (30, 35),
    # 16: (35, 40),
    # 17: (40, 45),
    # 18: (45, 50),
    # 19: (50, 60),
    # 20: (60, 80),
    # 21: (80, 10000),
    val_reg_list = []
    for v in value_list:
        val_reg = 0
        if v == 0:
            val_reg = 1
        elif v == 1:
            val_reg = 3
        elif v == 2:
            val_reg = 5
        elif v == 3:
            val_reg = 7
        elif v == 4:
            val_reg = 9
        elif v == 5:
            val_reg = 11
        elif v == 6:
            val_reg = 13
        elif v == 7:
            val_reg = 15
        elif v == 8:
            val_reg = 17
        elif v == 9:
            val_reg = 19
        elif v == 10:
            val_reg = 21
        elif v == 11:
            val_reg = 23
        elif v == 12:
            val_reg = 25
        elif v == 13:
            val_reg = 27
        elif v == 14:
            val_reg = 29
        elif v == 15:
            val_reg = 32.5
        elif v == 16:
            val_reg = 37.5
        elif v == 17:
            val_reg = 42.5
        elif v == 18:
            val_reg = 47.5
        elif v == 19:
            val_reg = 55
        elif v == 20:
            val_reg = 70
        elif v == 21:
            val_reg = 90
        val_reg_list.append(val_reg)
    return val_reg_list


def save_pred_res(v_true, v_pred, fn):
    df_pred_res = pd.DataFrame()
    df_pred_res['true'] = v_true
    df_pred_res['pred'] = v_pred
    df_pred_res.to_csv(fn, index=False)


def save_pred_res_mapping(v_pred, fn):
    df_pred_res = pd.DataFrame()
    df_pred_res['pnt_id'] = np.arange(0, len(v_pred), 1)
    df_pred_res['pred'] = v_pred
    df_pred_res.to_csv(fn, index=False)
