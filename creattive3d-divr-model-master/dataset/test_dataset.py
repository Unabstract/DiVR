import torch.utils.data as data
import torch
import numpy as np
import os
import json
import pickle
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data
from utils.dataset_utils import *


class GustTestDataset(data.Dataset):
    def __init__(self, config, train=False):

        self.config = config
        self.train = train
        self.dataroot = config.dataroot
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        self.fps = config.fps
        self.data_freq = config.data_freq
        self.disable_gaze = config.disable_gaze
        self.train_set = config.train_set

        self.dataset_info = pd.read_csv(os.path.join(self.dataroot, self.train_set))
        # print(self.dataset_info)
        self.parse_data_info()
        self.load_scene()

        self.random_ori_list = [-180, -90, 0, 90]

    def __getitem__(self, index):
        ego_idx = self.poses_path_list[index]
        scene = self.scenes_path_list[index]
        seq = self.sequences_path_list[index]
        task = self.task_list[index]
        start_frame, end_frame = self.start_end_list[index]

        poses_input_idx = []
        gazes = []
        node_features_data = []
        edge_index_data = []
        edge_features_data = []
        gazes_mask = []
        poses_input = []
        smplx_vertices = []
        random_ori = np.random.choice(self.random_ori_list)
        random_rotation = Rotation.from_euler('xyz', [0, random_ori, 0], degrees=True).as_matrix()

        transform_path = self.trans_path_list[index]
        transform_info = json.load(open(os.path.join(self.dataroot, scene, seq, transform_path), 'r'))
        scale = transform_info['scale']
        trans_pose2scene = np.array(transform_info['transformation'])
        trans_pose2scene[:3, 3] /= scale
        transform_norm = np.loadtxt(os.path.join(self.dataroot, scene, 'scene_obj', 'transform_norm.txt')).reshape(
            (4, 4))
        transform_norm[:3, 3] /= scale
        # trans_scene2pose = np.linalg.inv(trans_pose2scene)
        transform_pose = transform_norm @ trans_pose2scene

        for f in range(self.input_seq_len):
            pose_idx = ego_idx + int(f * self.data_freq / self.fps)
            poses_input_idx.append(pose_idx)

            gaze_points = np.zeros((1, 3))  # (self.config.gaze_points, 3))
            gazes_mask.append(torch.zeros(1).long())

            # hetero_graph
            graph_path = os.path.join(self.dataroot, scene, seq, 'graph_data_het', f'g_{pose_idx}.json')
            if os.path.exists(graph_path):
                graph_metad = json.load(open(graph_path, 'r'))


                node_features = torch.tensor(graph_metad['node_features'], dtype=torch.float)
                edge_index = torch.tensor(graph_metad['edge_index'], dtype=torch.long)
                edge_features = torch.tensor(graph_metad['edge_features'], dtype=torch.float)

                node_features_data.append(node_features)
                edge_index_data.append(edge_index)
                edge_features_data.append(edge_features)


            if not self.disable_gaze:
                gaze_ply_path = os.path.join(self.dataroot, scene, seq, 'eye_pc',
                                             '{}_center.ply'.format(pose_idx))

                if os.path.exists(gaze_ply_path):

                    gaze_data = trimesh.load_mesh(gaze_ply_path)
                    gaze_data.apply_scale(1 / scale)
                    gaze_data.apply_transform(transform_norm)

                    points = gaze_data.vertices
                    if np.sum(abs(points)) > 1e-8:
                        gazes_mask[-1] = torch.ones(1).long()
                    gaze_points = gaze_data.vertices[0:1]

            pose_data = pickle.load(open(os.path.join(self.dataroot, scene, seq, 'motion',
                                                      '{}.pkl'.format(pose_idx)), 'rb'))

            ori = pose_data['orient'].detach().cpu().numpy()
            trans = pose_data['trans'].detach().cpu().numpy().reshape((3, 1))
            R = Rotation.from_rotvec(ori).as_matrix()

            R_s = transform_pose[:3, :3] @ R
            ori_s = Rotation.from_matrix(R_s).as_rotvec()
            trans_s = (transform_pose[:3, :3] @ trans + transform_pose[:3, 3:]).reshape(3)

            if self.train:
                ori_s = Rotation.from_matrix(random_rotation @ R_s).as_rotvec()
                trans_s = (random_rotation @ trans_s.reshape((3, 1))).reshape(3)

                gaze_points = (random_rotation @ gaze_points.T).T

            poses_input.append(torch.cat([torch.from_numpy(ori_s.copy()).float(),
                                          torch.from_numpy(trans_s.copy()).float(),
                                          pose_data['latent']]))

            gazes.append(torch.from_numpy(gaze_points).float())

            smplx = trimesh.load_mesh(os.path.join(self.dataroot, scene, seq, 'motion',
                                                   '{}.obj'.format(pose_idx)))

            smplx_vertices.append(torch.from_numpy(smplx.vertices).float())

        # graph: node features and edge indices for 6 timestamps
        if len(node_features_data) != 0:
            num_nodes = node_features.shape[0]
            node_features = {f'x_t{i}': node_features_data[i] for i in range(self.config.input_seq_len)}
            edge_indices = {f'edge_index_t{i}': edge_index_data[i] for i in range(self.config.input_seq_len)}

            # Creating SceneData instance with the prepared data
            edge_features = {f'edge_feat_t{i}': edge_features_data[i] for i in range(self.config.input_seq_len)}
            graphs = SceneData_het(num_nodes=num_nodes, **node_features, **edge_indices, **edge_features)

        gazes = torch.stack(gazes, dim=0)

        poses_input = torch.stack(poses_input, dim=0).detach()
        gazes_mask = torch.stack(gazes_mask, dim=0)

        gazes_valid_id = torch.where(gazes_mask)
        gazes_invalid_id = torch.where(torch.abs(gazes_mask - 1))
        gazes_valid = gazes[gazes_valid_id]
        gazes[gazes_invalid_id] *= 0
        gazes[gazes_invalid_id] += torch.mean(gazes_valid, dim=0, keepdim=True)
        smplx_vertices = torch.stack(smplx_vertices, dim=0)

        mask = []
        poses_label = []
        poses_predict_idx = []
        for f in range(self.output_seq_len + 1):
            pose_idx = ego_idx + int(self.input_seq_len * self.data_freq / self.fps) + int(f * self.data_freq / self.fps)
            poses_predict_idx.append(pose_idx)
            pose_path = os.path.join(self.dataroot, scene, seq, 'motion',
                                     '{}.pkl'.format(pose_idx if f < self.output_seq_len else end_frame))

            if not os.path.exists(pose_path) or pose_idx >= end_frame:
                poses_label.append(poses_label[-1])
                mask.append(torch.zeros(1).float())

            else:
                pose_data = pickle.load(open(pose_path, 'rb'))

                ori = pose_data['orient'].detach().cpu().numpy()
                trans = pose_data['trans'].detach().cpu().numpy().reshape((3, 1))
                R = Rotation.from_rotvec(ori).as_matrix()
                R_s = transform_pose[:3, :3] @ R
                ori_s = Rotation.from_matrix(R_s).as_rotvec()
                trans_s = (transform_pose[:3, :3] @ trans + transform_pose[:3, 3:]).reshape(3)

                if self.train:
                    ori_s = Rotation.from_matrix(random_rotation @ R_s).as_rotvec()
                    trans_s = (random_rotation @ trans_s.reshape((3, 1))).reshape(3)

                poses_label.append(
                    torch.cat([torch.from_numpy(ori_s.copy()).float(), torch.from_numpy(trans_s.copy()).float(),
                               pose_data['latent']]))

                mask.append(torch.ones(1).float())
        poses_label = torch.stack(poses_label, dim=0).detach()
        poses_mask = torch.cat(mask)

        scene_points = self.scene_list['{}_{}'.format(seq, start_frame)]
        scene_points = scene_points[np.random.choice(range(len(scene_points)), self.config.sample_points)]
        scene_points *= 1 / scale
        scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T
        if self.train:
            scene_points = (random_rotation @ scene_points.T).T
            scene_points += np.random.normal(loc=0, scale=self.config.sigma, size=scene_points.shape)

        return gazes, gazes_mask, poses_input, smplx_vertices, poses_label, poses_mask, \
               torch.from_numpy(scene_points).float(), seq, scene, poses_predict_idx, poses_input_idx, task, graphs


    def __len__(self):
        return len(self.poses_path_list)

    def parse_data_info(self):
        self.sequences_path_list = []
        self.task_list = []
        self.scenes_path_list = []
        self.trans_path_list = []
        self.poses_path_list = []
        self.start_end_list = []
        for i, seq in enumerate(self.dataset_info['sequence_path']):
            if self.dataset_info['training'][i] != self.train:
                continue
            start_frame = self.dataset_info['start_frame'][i]
            end_frame = self.dataset_info['end_frame'][i]
            scene = self.dataset_info['scene'][i]
            task = self.dataset_info['task'][i]
            transform = self.dataset_info['transformation'][i]
            # print(start_frame, end_frame, seq)

            end_frame = self.dataset_info['end_frame'][i] \
                        - int((self.input_seq_len + self.output_seq_len) * self.data_freq / self.fps)

            total_win = end_frame - start_frame
            if total_win < self.config.slide_win_eval:
                seq_num = 1
            else:
                seq_num = total_win // self.config.slide_win_eval
            for j in range(seq_num):
                k = start_frame + self.config.slide_win * j
                self.poses_path_list.append(k)
                self.sequences_path_list.append(seq)
                self.task_list.append(task)
                self.scenes_path_list.append(scene)
                self.trans_path_list.append(transform)
                self.start_end_list.append([self.dataset_info['start_frame'][i], self.dataset_info['end_frame'][i]])

    def load_scene(self):
        self.scene_list = {}
        for i, seq in enumerate(self.dataset_info['sequence_path']):
            if self.dataset_info['training'][i] != self.train:
                continue
            print('loading scene of {}'.format(seq))
            scene = self.dataset_info['scene'][i]
            start_frame = self.dataset_info['start_frame'][i]
            scene_ply = trimesh.load_mesh(os.path.join(self.dataroot, scene, 'scene_obj', 'scene_downsampled.ply'))
            scene_points = scene_ply.vertices

            self.scene_list['{}_{}'.format(seq, start_frame)] = scene_points