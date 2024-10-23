# coding=utf-8
import os

# model hyper-parameters
device = 'cpu'  # 'cpu' or 'cuda'
rand_seed = 314

model_name = 'DL'

# hyper-parameter of deep learning model
num_channels_band = 7
num_channels_topo = 2
num_channels_climate = 19
num_channels_vege = 3
num_channels_bedrock = 1

# hyper-parameter for training
lr = 1e-3
batch_size = 32
batch_size_pred = 10240
epochs = 1000
eval_interval = 1

study_area = 'Global'  # Global, Anhui, Switzerland, Denmark, Australia

data_dir = '../data/'
log_dir = './log/'
use_tensorboard = False

f_df_samples = os.path.join(data_dir, 'DL_inputs/global/samples/df_soc_0to30.csv')
# f_df_samples = os.path.join(data_dir, 'samples/samples_{}_processed.csv'.format(study_area))
# f_df_samples = os.path.join(data_dir, 'samples/samples_{}_allloc.csv'.format(study_area))

f_df_cov_stats = os.path.join(data_dir, 'DL_inputs/global/samples/df_var_stats.csv')

target_var_name_reg = 'soc'
target_var_name_cls = 'soc'

dir_data_x = os.path.join(data_dir, 'DL_inputs/global/')
# dir_data_x = os.path.join(data_dir, 'DL_inputs/regional/{}/'.format(study_area))
# dir_data_x = os.path.join(data_dir, 'DL_inputs/regional/{}/_whole_area/'.format(study_area))

num_class = 22

var_names_band = ['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7']
var_names_topo = ['elevation', 'slope']
var_names_climate = ['bio01', 'bio02', 'bio03', 'bio04', 'bio05', 'bio06', 'bio07', 'bio08', 'bio09', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19']
# var_names_climate = ['bio01', 'bio12']
var_names_vege = ['evi_mean', 'evi_min', 'evi_max']
var_names_bedrock = ['bedrock']

win_size_radius_band = 0
win_size_radius_topo = 0
win_size_radius_climate = 0
win_size_radius_vege = 0
win_size_radius_bedrock = 0

f_train_index = os.path.join(data_dir, 'DL_inputs/global/samples/data_split/', 'train.csv')
f_valid_index = os.path.join(data_dir, 'DL_inputs/global/samples/data_split/', 'valid.csv')
f_test_index = os.path.join(data_dir, 'DL_inputs/global/samples/data_split/', 'test.csv')

# use_refined_samples = False
# refined_area = 'Anhui'
# f_refined_samples_index = os.path.join(data_dir, 'DL_inputs/global/samples/data_split/', 'df_global_selected_for_regional_refine_{}.csv'.format(refined_area))

# f_train_index = os.path.join(data_dir, 'DL_inputs/regional/{}/samples/data_split'.format(study_area), 'train.csv')
# f_valid_index = os.path.join(data_dir, 'DL_inputs/regional/{}/samples/data_split'.format(study_area), 'test.csv')
# f_test_index = os.path.join(data_dir, 'DL_inputs/regional/{}/samples/data_split'.format(study_area), 'test.csv')

ex_id = 1
model_save_pth = './model/{}_{}_{}.pth'.format(model_name, study_area, ex_id)

use_pretrain_model = True
pretrain_model_name = 'GSoilCPM'
pretrain_model_save_pth = './model/pretrain_model/{}.pth'.format(pretrain_model_name)

save_predictions = False
DL_model_level = 1
fn_pred_res = os.path.join(data_dir, 'results/df_pred_res_DLp_{}_{}.csv'.format(DL_model_level, study_area))

save_predictions_mapping = False
fn_pred_res_mapping = os.path.join(data_dir, 'results/df_pred_res_mapping_DLp_{}.csv'.format(study_area))
