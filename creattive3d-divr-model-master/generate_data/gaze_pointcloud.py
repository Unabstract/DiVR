# usage: python3 ./generate_data/gaze_pointcloud.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train --csv_file dataset_model4.csv --slide_win 30

import open3d as o3d
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse

pd.options.mode.chained_assignment = None

# os.chdir('.')
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

parser.add_argument('--csv_file',
                    type=str,
                    required=True,
                    help='Path to the csv file from which we generate the data')

parser.add_argument('--slide_win',
                    type=int,
                    required=True,
                    help='Sliding window for frames'
                    )


args = parser.parse_args()

data_path = Path(args.data_path)
results_path = Path(args.results_path)
slide_win = args.slide_win

if not results_path.exists():
    results_path.mkdir(parents=True)

dataset_path = results_path / args.csv_file
df_dataset = pd.read_csv(dataset_path)

for idx in tqdm(df_dataset.index, leave=False):

    notation = df_dataset.loc[idx]['sequence_path']
    start_frame = df_dataset.loc[idx]['start_frame']
    end_frame = df_dataset.loc[idx]['end_frame']

    print(f'processing: {notation}')
    r, c, s = notation.split('_')

    results_folder = results_path / f'{s}' / f'{notation}' / 'eye_pc'
    print(results_folder)
    if not results_folder.exists():
        results_folder.mkdir(parents=True)

    POR_file = data_path / f'{r}' / f'{c}' / f'{s}' / f'{notation}_POR.json'

    try:
        with open(POR_file, 'r') as json_file:
            json_load = json.load(json_file)
    except FileNotFoundError:
        print(f'File missing: {POR_file}')
        continue

    data = pd.DataFrame(json_load)
    coord = pd.DataFrame(data['centroid'].str[0].to_dict()).iloc[0:3].T
    centroid_df = pd.concat([coord, data['timestamp'].str[0]], axis=1)
    centroid_df = centroid_df.loc[:, ['timestamp', 'x', 'y', 'z']]
    centroid_df.columns = ['unixTime', 'x', 'y', 'z']

    centroid_df = centroid_df.dropna()
    centroid_df = centroid_df.set_index('unixTime')
    centroid_df['z'] = centroid_df['z'] * -1
    # interpolation 60Hz
    fixed_frequency_time = np.arange(centroid_df.index[0], centroid_df.index[-1], 1000/60)
    x_60 = np.interp(fixed_frequency_time, centroid_df.index.astype('float64'), centroid_df['x'].astype('float64'))
    y_60 = np.interp(fixed_frequency_time, centroid_df.index.astype('float64'), centroid_df['y'].astype('float64'))
    z_60 = np.interp(fixed_frequency_time, centroid_df.index.astype('float64'), centroid_df['z'].astype('float64'))
    centroid_df_60 = pd.concat([pd.Series(x_60), pd.Series(z_60), pd.Series(y_60)], axis=1)
    centroid_df_60.columns = ['x', 'z', 'y']

    total_win = end_frame - start_frame
    if total_win < slide_win:
        seq_num = 1
    else:
        seq_num = total_win // slide_win

    for i in tqdm(range(seq_num)):
        index_ord = start_frame + slide_win*i
        if index_ord < centroid_df_60.index.size:
            index_data = centroid_df_60.index[index_ord]
            xyz = np.array(centroid_df_60.loc[index_data,['x', 'z', 'y']])
            xyz = xyz.reshape(1,-1)
            if np.isnan(xyz).any():
                print(f'NaN values detected: {notation}_{index_ord}')
                break
            else:
                # print(xyz)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                o3d.io.write_point_cloud(results_path /f'{s}'/f'{notation}'/'eye_pc'/f'{index_ord}_center.ply', pcd)
