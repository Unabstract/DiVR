#usage: python3 motion_data.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train_new/ --csv_file dataset_model4.csv --slide_win 30

from scipy.spatial.transform import Rotation as R
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import pickle


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

    parser.add_argument('--slide_win',
                        type=int,
                        required=True,
                        help='Sliding window for frames'
                        )

    args = parser.parse_args()

    # data
    data_path = Path(args.data_path)
    results_path = Path(args.results_path)
    if not results_path.exists():
        results_path.mkdir(parents=True)

    dataset_csv = results_path / args.csv_file
    df_dataset = pd.read_csv(dataset_csv)

    motion_data = {}
    for idx in tqdm(df_dataset.index, leave=False):
        notation = df_dataset.loc[idx]['sequence_path']
        start_frame = df_dataset.loc[idx]['start_frame']
        end_frame = df_dataset.loc[idx]['end_frame']

        print(f'processing: {notation}')
        r, c, s = notation.split('_')

        motion_path = results_path / f'{s}' / f'{notation}' / 'c'

        print(motion_path)
        if not motion_path.exists():
            motion_path.mkdir(parents=True)

        data_folder = data_path / f'{r}' / f'{c}' / f'{s}'
        csv_file_pos = data_folder / f'{notation}_motion_pos.csv'
        csv_file_rot = data_folder / f'{notation}_motion_rot.csv'
        df_csv_pos = pd.read_csv(csv_file_pos)
        df_csv_rot = pd.read_csv(csv_file_rot)

        # computing absolute angles for head joint from bvh data
        joint_hierarchy = ["Hips", "Chest", "Chest2", "Chest3", "Chest4", "Neck", "Head"]
        joint_x = [s + '.x' for s in joint_hierarchy]
        joint_y = [s + '.y' for s in joint_hierarchy]
        joint_z = [s + '.z' for s in joint_hierarchy]
        df_csv_rot['sum.x'] = df_csv_rot[joint_x].sum(axis=1)
        df_csv_rot['sum.y'] = df_csv_rot[joint_y].sum(axis=1)
        df_csv_rot['sum.z'] = df_csv_rot[joint_z].sum(axis=1)

        total_win = end_frame - start_frame
        if total_win < args.slide_win:
            seq_num = 1
        else:
            seq_num = total_win // args.slide_win

        for i in tqdm(range(seq_num)):
            index_f = start_frame + args.slide_win*i

            euler_angles = df_csv_rot.loc[index_f, ['sum.y', 'sum.x', 'sum.z']].values

            r = R.from_euler('yxz', euler_angles, degrees=True)
            orient_vec = r.as_rotvec(degrees=False)
            orient_vec = torch.tensor(orient_vec, dtype = torch.float32)

            trans_vec = torch.tensor(df_csv_pos.loc[index_f, ['Head.x', 'Head.y', 'Head.z']].values, dtype=torch.float32)
            motion_data['orient'] = orient_vec
            motion_data['trans'] = trans_vec

            with open(motion_path / f'{index_f}.pkl', 'wb') as f:
                pickle.dump(motion_data, f)




