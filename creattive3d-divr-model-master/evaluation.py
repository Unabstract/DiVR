import matplotlib.image

from dataset import test_dataset
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import time
import torch.nn.functional as F
from utils.metrics import *
from config.config import DiVRConfig
from model.DiVR_het import DiVR_het
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

np.random.seed(42)

class motion_evaluator():
    def __init__(self, config):
        self.config = config
        self.test_dataset = test_dataset.GustTestDataset(config, train=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_path = os.path.join(self.config.dataroot)


    def eval_sequence(self, model=None):
        # Load model if not provided
        if model is None:
            if config.model_type == 'DiVR_het':
                model = DiVR_het(config)
            else:
                raise NotImplementedError
            model = model.to(self.device)
            assert self.config.load_model_dir is not None
            print('loading pretrained model from ', self.config.load_model_dir)
            # model.load_state_dict(torch.load(self.config.load_model_dir, map_location=self.device))
            model.load_state_dict(torch.load(self.config.load_model_dir))
            print('load done!')
        with torch.no_grad():
            model.eval()

            # creating dict of same seq and task, with different start_frame
            dict_seq_task = {}
            print('Collecting data')
            for n_seq, data in tqdm(enumerate(self.test_dataset)):

                gazes, gazes_mask, poses_input, smplx_vertices, poses_label, poses_mask, scene_points, seq, scene, \
                poses_predict_idx, poses_input_idx, task, graphs = data

                if f'{seq}_{task}' not in dict_seq_task.keys():
                    dict_seq_task[f'{seq}_{task}'] = []
                dict_seq_task[f'{seq}_{task}'].append(n_seq)

            all_metrics = []
            sigma_val = 1
            radius_val = 1

            for key_t in tqdm(dict_seq_task.keys()):
                print(f'Evaluating sequence: {key_t}')
                seq_task_index = dict_seq_task[key_t]
                total_seq = len(seq_task_index)
                list_keys = []
                total_n_seq = 10
                for i in range(total_seq + total_n_seq):
                    list_keys.append('f' + str(i))

                save_path = os.path.join(self.config.output_path)
                os.makedirs(save_path, exist_ok=True)

                save_data_path = os.path.join(save_path, 'data')
                os.makedirs(save_data_path, exist_ok=True)

                # Initialize metric storage
                mse_seq_task = []
                od_seq_task = []
                ade_seq_task = []
                fde_seq_task = []
                pos_dict_pred = {key: [] for key in list_keys}
                pos_dict_gt = {key: [] for key in list_keys}
                ori_dict_pred = {key: [] for key in list_keys}
                ori_dict_gt = {key: [] for key in list_keys}
                vel_dict_pred = {key: [] for key in list_keys[:-2]}
                vel_dict_gt = {key: [] for key in list_keys[:-2]}
                ang_vel_dict_pred = {key: [] for key in list_keys[:-2]}
                ang_vel_dict_gt = {key: [] for key in list_keys[:-2]}
                mse_dict = {key: [] for key in list_keys}
                od_dict = {key: [] for key in list_keys}
                pos_ori_vis_dict = {}
                cte = 0

                for n_seq in tqdm(seq_task_index):

                    gazes, gazes_mask, poses_input, smplx_vertices, poses_label, poses_mask, scene_points, seq, scene, \
                    poses_predict_idx, poses_input_idx, task, graphs = self.test_dataset[n_seq]

                    # Preparing input data for the model
                    gazes = gazes.unsqueeze(0).to(self.device)
                    gazes_mask = gazes_mask.unsqueeze(0).to(self.device)
                    poses_mask = poses_mask.unsqueeze(0).to(self.device)
                    poses_input = poses_input.unsqueeze(0).to(self.device)
                    smplx_vertices = smplx_vertices.unsqueeze(0).to(self.device)
                    poses_label = poses_label.unsqueeze(0).to(self.device)
                    scene_points = scene_points.unsqueeze(0).to(self.device).contiguous()
                    graphs = graphs.to(self.device)

                    poses_predict = model(gazes, gazes_mask, poses_input, smplx_vertices, scene_points, graphs)

                    # Prepare data for evaluation
                    input_pos = poses_input[:, :, 3:6].detach().cpu()
                    input_ori = rotation_vectors_to_quaternions(poses_input[:, :, :3].detach().cpu())
                    input_pos = F.pad(input_pos, (0, 0, 0, 4))
                    input_ori = F.pad(input_ori, (0, 0, 0, 4))
                    gt_pos = poses_label[:, :-1, 3:6].detach().cpu()
                    gt_ori = rotation_vectors_to_quaternions(poses_label[:, :-1, :3].detach().cpu())

                    pred_pos = poses_predict[:, self.config.input_seq_len:-1, 3:6].detach().cpu()
                    pred_ori = rotation_vectors_to_quaternions(poses_predict[:, self.config.input_seq_len:-1, :3].detach().cpu())

                    # Calculate metrics
                    mse_loss = F.mse_loss(pred_pos, gt_pos, reduction='none').squeeze(0).mean(-1).cpu().numpy()
                    orth_distance = great_circle_distance(pred_ori, gt_ori).squeeze(0).cpu().numpy()
                    mse_loss_seq =  np.mean(mse_loss)
                    od_seq = np.mean(orth_distance)
                    ade_seq = compute_ade(pred_pos, gt_pos)
                    fde_seq = compute_fde(pred_pos, gt_pos)

                    mse_seq_task.append(mse_loss_seq)
                    od_seq_task.append(od_seq)
                    ade_seq_task.append(ade_seq)
                    fde_seq_task.append(fde_seq)

                    local_keys = list_keys[cte:cte + total_n_seq]

                    # Saving position and orientation
                    for key, val in zip(local_keys, gt_pos.squeeze(0).cpu().tolist()):
                        pos_dict_gt[key].append(val)
                    for key, val in zip(local_keys, gt_ori.squeeze(0).cpu().tolist()):
                        ori_dict_gt[key].append(val)
                    for key, val in zip(local_keys, pred_pos.squeeze(0).cpu().tolist()):
                        pos_dict_pred[key].append(val)
                    for key, val in zip(local_keys, pred_ori.squeeze(0).cpu().tolist()):
                        ori_dict_pred[key].append(val)

                    # Saving metrics
                    for key, val in zip(local_keys, list(mse_loss)):
                        mse_dict[key].append(np.float64(val))
                    for key, val in zip(local_keys, list(orth_distance)):
                        od_dict[key].append(np.float64(val))

                    # Velocity dataframes
                    pos_df = torch.cat([gt_pos.squeeze(0), pred_pos.squeeze(0)], 1)
                    pos_df = pd.DataFrame(pos_df.cpu().numpy())
                    pos_df.columns = ['x_gt', 'y_gt', 'z_gt', 'x_p', 'y_p', 'z_p']

                    vel_df = pos_df.diff()
                    vel_df = vel_df.iloc[1:]
                    vel_df['vel_gt'] = np.linalg.norm(vel_df.iloc[:,0:3], axis=1) / (self.config.slide_win_eval / self.config.data_freq)
                    vel_df['vel_pred'] = np.linalg.norm(vel_df.iloc[:, 3:6], axis=1) / (self.config.slide_win_eval / self.config.data_freq)

                    vel_input = vel_df['vel_gt']
                    vel_input = pd.Series(gaussian_filter1d(vel_input, sigma=sigma_val, radius=radius_val))
                    vel_pred = vel_df['vel_pred']
                    vel_pred = pd.Series(gaussian_filter1d(vel_pred, sigma=sigma_val, radius=radius_val))

                    for key, val in zip(local_keys[:-1], vel_pred):
                        vel_dict_pred[key].append(np.float64(val))

                    for key, val in zip(local_keys[:-1], vel_input):
                        vel_dict_gt[key].append(np.float64(val))

                    export_df = torch.cat([gt_pos.squeeze(0), gt_ori.squeeze(0),
                                           pred_pos.squeeze(0), pred_ori.squeeze(0),
                                           input_pos.squeeze(0), input_ori.squeeze(0)], 1)

                    pos_ori_vis_dict[f'{seq}_{poses_input_idx[0]}'] = export_df.tolist()

                    # Angular velocity
                    gt_circle_dist = great_circle_distance(gt_ori[:, :-1, :], gt_ori[:, 1:, :]).squeeze(0).cpu().numpy()
                    gt_ang_vel = gt_circle_dist / (self.config.slide_win_eval / self.config.data_freq)
                    pred_circle_dist = great_circle_distance(pred_ori[:, :-1, :], pred_ori[:, 1:, :]).squeeze(0).cpu().numpy()
                    pred_ang_vel = pred_circle_dist / (self.config.slide_win_eval / self.config.data_freq)

                    gt_ang_vel = gaussian_filter1d(gt_ang_vel,  sigma=sigma_val, radius=radius_val)
                    pred_ang_vel = gaussian_filter1d(pred_ang_vel, sigma=sigma_val, radius=radius_val)

                    for key, val in zip(local_keys[:-1], list(gt_ang_vel)):
                        ang_vel_dict_gt[key].append(np.float64(val))

                    for key, val in zip(local_keys[:-1], list(pred_ang_vel)):
                        ang_vel_dict_pred[key].append(np.float64(val))

                    cte+=1

                # Save position and orientation visualization data
                with open(os.path.join(save_data_path, f'{seq}_vis_{poses_input_idx[0]}.json'), 'w') as json_file:
                    json.dump(pos_ori_vis_dict, json_file)

                # Calculate and save sequence metrics
                mse_all_seq = sum(mse_seq_task) / len(mse_seq_task)
                od_all_seq = sum(od_seq_task) / len(od_seq_task)
                ade_all_seq = sum(ade_seq_task) / len(ade_seq_task)
                fde_all_seq = sum(fde_seq_task) / len(fde_seq_task)
                all_metrics.append([f'{seq}_{poses_input_idx[0]-n_seq*30}', mse_all_seq, od_all_seq, ade_all_seq, fde_all_seq])


                all_data_dict = {'pos_dict_pred': pos_dict_pred, 'pos_dict_gt': pos_dict_gt,
                                 'ori_dict_pred':ori_dict_pred, 'ori_dict_gt': ori_dict_gt,
                                 'vel_dict_pred': vel_dict_pred, 'vel_dict_gt': vel_dict_gt,
                                 'ang_vel_dict_pred': ang_vel_dict_pred, 'ang_vel_dict_gt': ang_vel_dict_gt,
                                 'mse_dict': mse_dict, 'od_dict': od_dict}


                with open(os.path.join(save_data_path, f'{seq}_{poses_input_idx[0]}.json'), 'w') as json_file:
                    json.dump(all_data_dict, json_file)

                # Create plot directories
                save_plot_path = os.path.join(save_path, f'plots')
                os.makedirs(save_plot_path, exist_ok=True)

                # plot positions
                image_scene = os.path.join(self.image_path, f'{scene}', 'scene_obj', 'scene.png')
                img = matplotlib.image.imread(image_scene)

                for n_key, key in enumerate(pos_ori_vis_dict.keys()):

                    if '0V' in scene:
                        x_min, x_max, y_min, y_max = -5.0, 0.0, -4.0, 5.3
                        fig, ax = plt.subplots(figsize=(5, 9.3))
                    elif '1V' in scene:
                        x_min, x_max, y_min, y_max = -5.0, 5.0, -4.0, 5.3
                        fig, ax = plt.subplots(figsize=(10, 9.3))
                    elif '2V' in scene:
                        x_min, x_max, y_min, y_max = -12.0, 5.0, -3.0, 9.0
                        fig, ax = plt.subplots(figsize=(17, 12))
                    else:
                        print('Scene not valid!!')

                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    pos_ori_df = pd.DataFrame(pos_ori_vis_dict[key], columns=['x_gt', 'y_gt', 'z_gt',
                                                                              'q1_gt', 'q2_gt', 'q3_gt', 'q4_gt',
                                                                              'x_p', 'y_p', 'z_p',
                                                                              'q1_p', 'q2_p', 'q3_p', 'q4_p',
                                                                              'x_in', 'y_in', 'z_in',
                                                                              'q1_in', 'q2_in', 'q3_in', 'q4_in'
                                                                              ])

                    x_gt = pos_ori_df['x_gt']
                    z_gt = pos_ori_df['z_gt']
                    x_p = pos_ori_df['x_p']
                    z_p = pos_ori_df['z_p']
                    x_in = pos_ori_df['x_in'][:6]
                    z_in = pos_ori_df['z_in'][:6]

                    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto')
                    ax.plot(x_in, z_in, "r-", linewidth=3.2, label='input')
                    ax.plot(x_in.iloc[-1], z_in.iloc[-1], "og", markersize=8, markeredgewidth=5)
                    ax.plot(x_p, z_p, "m--", linewidth=3.4, label='pred')
                    ax.plot(x_gt, z_gt, "b-", linewidth=3.2, label='gt')

                    ax.tick_params(axis='both', labelsize=20)
                    # ax.legend(fontsize=15)

                    fig_path = os.path.join(save_plot_path, f'pos_{key}')
                    plt.savefig(fig_path, bbox_inches='tight')
                    plt.close()

                # Plot velocities
                vel_median_list = []
                vel_median_list_gt = []
                vel_min_list = []
                vel_max_list = []

                for key in vel_dict_pred.keys():
                    vel_median = np.median(vel_dict_pred[key])
                    vel_median_gt = np.median(vel_dict_gt[key])
                    vel_median_list.append(vel_median)
                    vel_median_list_gt.append(vel_median_gt)
                    vel_error = vel_dict_pred[key] - vel_median
                    vel_min_list.append(np.min(vel_error))
                    vel_max_list.append(np.max(vel_error))

                plt.figure(figsize=(10,6))
                N = len(list_keys[:-2])
                x = np.arange(N)
                x = x * (self.config.slide_win_eval / self.config.data_freq)
                plt.plot(x, vel_median_list, 'r--', label='pred vel')
                plt.plot(x, vel_median_list_gt, 'b--', label='gt vel')

                fill_min = [vel_median_list[i] + vel_min_list[i] for i in range(len(vel_median_list))]
                fill_max = [vel_median_list[i] + vel_max_list[i] for i in range(len(vel_median_list))]

                plt.fill_between(x, fill_min, fill_max, color='r', alpha=0.2)
                plt.xlabel('time(s)')
                plt.ylabel('velocity(m/s)')
                plt.ylim(0.0, 2.0)
                plt.title(f'{key_t} mse:{mse_all_seq}')
                plt.legend()
                plt.savefig(os.path.join(save_plot_path, 'vel_{}_{}.png'.format(seq, poses_input_idx[0])),
                            bbox_inches='tight')
                plt.close()

                # angular velocities
                vel_median_list = []
                vel_median_list_gt = []
                vel_min_list = []
                vel_max_list = []

                for key in ang_vel_dict_pred.keys():
                    if len(ang_vel_dict_pred[key]) > 1:
                        vel_median = np.median(ang_vel_dict_pred[key])
                        vel_median_gt = np.median(ang_vel_dict_gt[key])
                        vel_median_list.append(vel_median)
                        vel_median_list_gt.append(vel_median_gt)
                        vel_error = ang_vel_dict_pred[key] - vel_median
                        vel_min_list.append(np.min(vel_error))
                        vel_max_list.append(np.max(vel_error))
                    elif len(ang_vel_dict_pred[key]) == 1:
                        uniq_pred_vel = ang_vel_dict_pred[key][0]
                        uniq_gt_vel = ang_vel_dict_gt[key][0]
                        vel_median_list.append(uniq_pred_vel)
                        vel_median_list_gt.append(uniq_gt_vel)
                        vel_min_list.append(0)
                        vel_max_list.append(0)
                    else:
                        continue

            # Save all metrics
            df_metrics = pd.DataFrame(all_metrics, columns=['seq', 'mse', 'od', 'ade', 'fde'])
            df_metrics.to_csv(os.path.join(save_data_path, f'metrics_test.csv'), index=False)

            # Print test results
            print(f"ADE: {df_metrics['ade'].mean()}")
            print(f"FDE: {df_metrics['fde'].mean()}")

if __name__ == '__main__':
    config = DiVRConfig().parse_args()
    start = time.time()
    evaluator = motion_evaluator(config)
    r = evaluator.eval_sequence()

