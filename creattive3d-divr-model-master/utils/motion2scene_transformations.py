# usage: python3 ./gen_gaze_dataset/motion2scene_transformations.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train

import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
import json
import os
import warnings
from scipy.spatial.transform import Rotation as R

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def rigid_transform_3D(A, B):
    '''
    Funtion to compute rotation and translation transformation using Procrustes analysis.
    :param A: set A
    :param B: set B
    :return: Rotation matrix and translation vector
    '''
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


# Function to compute correlation
def correlation(x, y):
    '''
    Find correlation values between two time series data
    :param x:
    :param y:
    :return:
    '''
    shortest = min(x.shape[0], y.shape[0])

    return np.corrcoef(x.iloc[:shortest].values, y.iloc[:shortest].values)[0, 1]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='Path to the GUsT3D data folder'
                        )
    parser.add_argument('--results_path',
                        type=str,
                        required=True,
                        help='Path to the folder in which we are going to generate the data for training'
                        )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    train_path = Path(args.results_path)

    if not train_path.exists():
        train_path.mkdir(parents=True, exist_ok=True)

    quest_path = data_path / 'questionnaire.csv'
    quest_df = pd.read_csv(quest_path, sep=';', encoding='latin-1')
    subject_list = quest_df.loc[:39, 'uid']

    condition_list = ['RWNV', 'RWLV']
    scene_list = ['CI0V', 'CI1V', 'CI2V', 'SI0V', 'SI1V', 'SI2V']

    df_corr_all = pd.DataFrame(columns=['corr_value'])
    columns_transf = ['rot_y', 'rot_x', 'rot_z', 'transl_x', 'transl_y', 'transl_z']

    df_trans_val_bvh2gust = pd.DataFrame(columns=columns_transf)
    df_trans_val_bvh2gust_approx = pd.DataFrame(columns=columns_transf)
    df_trans_val_gust2bvh = pd.DataFrame(columns=columns_transf)
    df_trans_val_gust2bvh_approx = pd.DataFrame(columns=columns_transf)

    for subject in tqdm(subject_list):
        results_path = Path(f'Results/R{subject}')
        for condition in condition_list:
            for scene in tqdm(scene_list, leave=False):

                notation = f'U{int(subject):03d}_{condition}_{scene}'
                print(f'{notation}')
                gust_pos_file = data_path / f'U{int(subject):03d}' / f'{condition}' / f'{scene}' / f'{notation}_POR.json'
                bvh_pos_file = data_path / f'U{int(subject):03d}' / f'{condition}' / f'{scene}' / f'{notation}_motion_pos.csv'

                try:
                    with open(gust_pos_file, 'r') as json_file:
                        json_load = json.load(json_file)
                except FileNotFoundError:
                    print(f'File missing: {gust_pos_file}')
                    continue
                data = pd.DataFrame(json_load)

                df_gust_pos = pd.concat([pd.DataFrame.from_records(data['origin'].str[0]), data['timestamp'].str[0]],
                                      axis=1)

                df_gust_pos = df_gust_pos.loc[:, ['timestamp', 'x', 'y', 'z']]
                df_gust_pos.columns = ['unixTime', 'x', 'y', 'z']

                index_gust = df_gust_pos.iloc[:-1, 0].values
                df_gust_pos = df_gust_pos.set_index('unixTime')
                df_gust_pos.columns = ['pH_x','pH_y','pH_z']

                df_gust_pos['pH_z'] = df_gust_pos['pH_z']*-1

                try:
                    df_bvh_pos = pd.read_csv(bvh_pos_file)
                except FileNotFoundError:
                    print(f'File not found: {bvh_pos_file}')
                    continue

                df_bvh_pos = df_bvh_pos.loc[:, ['Head.x', 'Head.y', 'Head.z']]

                # pos
                df_bvh_pos = df_bvh_pos.set_index(np.arange(index_gust[0], df_bvh_pos.index.size * 1000/60 + index_gust[0] + 300, 1000/60)[:df_bvh_pos.index.size])
                df_bvh_pos.columns = ['pM_x', 'pM_y', 'pM_z']

                # removing duplicate indexes
                df_gust_pos = df_gust_pos[~df_gust_pos.index.duplicated(keep="first")]

                # pos
                df_comb = pd.concat([df_bvh_pos, df_gust_pos], axis=1)
                df_comb = df_comb.sort_index()
                df_comb = df_comb.interpolate(method='index')

                # computing transformation from bvh to gust
                R_mat_bvh2gust, t_bvh2gust = rigid_transform_3D(np.array(df_comb.loc[:, ['pM_x', 'pM_y', 'pM_z']]).T,
                                                                np.array(df_comb.loc[:, ['pH_x','pH_y','pH_z']]).T)

                # saving rotation - angle values from bvh to gust
                euler_bvh2gust = R.from_matrix(R_mat_bvh2gust).as_euler('yxz', degrees=True)
                euler_bvh2gust_approx = euler_bvh2gust * [1, 0, 0]

                R_mat_bvh2gust_approx = R.from_euler('yxz', euler_bvh2gust_approx,  degrees=True).as_matrix()
                t_bvh2gust_approx = t_bvh2gust * [[1], [0], [1]]

                trans_mat_bvh2gust_approx = np.concatenate((np.concatenate((R_mat_bvh2gust_approx, t_bvh2gust_approx), axis=1),
                                                            np.array([0, 0, 0, 1]).reshape(1, -1)))

                trans_dict = {'scale': 1.0, 'transformation': trans_mat_bvh2gust_approx.tolist()}

                train_folder = train_path / f'{scene}' / f'{notation}'
                print(train_folder)

                if train_folder.exists():
                    with open(train_folder/'transform_info.json', "w") as outfile:
                        json.dump(trans_dict, outfile)





