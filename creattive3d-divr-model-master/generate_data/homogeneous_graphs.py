# usage: python ./graph_data/logs2graphs_f.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train_new --csv_file dataset_model4.csv --slide_win 30

import json
import argparse
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path

pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()

parser.add_argument('--data_path',
                    type=str,
                    required=True,
                    help='Path to the train data'
                    )

parser.add_argument('--results_path',
                    type=str,
                    required=True,
                    help='Path to the folder in which we are going to generate the graph data for training')

parser.add_argument('--csv_file',
                    type=str,
                    required=True,
                    help='Path to the csv file from which we generate the data')

parser.add_argument('--slide_win',
                    type=int,
                    required=True,
                    help='slide window to create graph data'
                    )

args = parser.parse_args()

# data_path = '../../../GUsT-3D/GustNewFormat/Data'
data_folder = Path(args.data_path) #

# results_path = '../../GUsT3D_data_train_new'
results_folder = Path(args.results_path)
train_path = results_folder / args.csv_file

# reading training file
train_df = pd.read_csv(train_path)

for index, row_train in tqdm(train_df.iterrows()):

    start_frame = row_train['start_frame']
    end_frame = row_train['end_frame']
    seq = row_train['sequence_path']
    subject, condition, scene = seq.split('_')
    end_utime = row_train['unixTime']
    end_datetime = pd.to_datetime(end_utime, unit='ms')
    start_utime = end_utime - row_train['total_frames'] * 1000 / 60
    start_datetime = pd.to_datetime(start_utime, unit='ms')

    # log file
    log_file = data_folder / f'{subject}' / f'{condition}' / f'{scene}' / f'{seq}_Logs.csv'

    # focusing on the relevant columns
    logs_df = pd.read_csv(log_file, usecols=['unixTime', 'localisation', 'light', 'button', 'carPosition'])

    # Convert unixTime
    logs_df['dateTime'] = pd.to_datetime(logs_df['unixTime'], unit='ms')
    logs_df = logs_df.set_index('dateTime')
    # Preliminary cleaning and preparation
    light_mapping = {'red': 1, 'orange': 2, 'green': 3}
    localisation_mapping = {'start_sidewalk': 'SW1','Crossing': 'R', 'Road': 'R', 'opposite_sidewalk': 'SW2',
                            'outside': 'SW1', 'House':'H'}
    # light
    logs_df['light'] = logs_df['light'].map(light_mapping)
    # localisation
    logs_df['localisation'] = logs_df['localisation'].map(localisation_mapping)

    # Determine if the button was pressed by the person
    logs_df['button'] = logs_df['button'].apply(lambda x: 1 if x == 'interact' else 0)

    # Check for vehicle presence
    logs_df['carPosition'] = logs_df['carPosition'].apply(lambda x: 1 if x != np.NaN else 0)
    # logs_df['vehiclePresent'] = logs_df['carPosition'].notna()

    # fill nan
    logs_df = logs_df.fillna(logs_df.mode().iloc[0])

    # sampling
    logs_df.drop('unixTime', axis=1, inplace=True)
    logs_df = logs_df.resample('16.66ms').ffill()
    logs_df = logs_df.iloc[1:]

    logs_df = logs_df[(logs_df.index >= start_datetime) & (logs_df.index <= end_datetime)]
    logs_df = logs_df.reset_index()
    logs_df.set_index(logs_df.index + row_train['start_frame'], inplace=True)

    graph_path = results_folder / f'{scene}' /f'{seq}'/ 'graph_data'

    if not graph_path.exists():
        Path.mkdir(graph_path, parents=True)

    slide_win = args.slide_win
    seq_num = (end_frame - start_frame) // slide_win  # +1

    # Extract unique localisations, lights, and car positions as nodes
    localisations = logs_df['localisation'].unique().tolist()
    car_exists = logs_df['carPosition'].unique().tolist()
    predefined_local = ['SW1', 'SW2', 'R', 'H']

    local_exists = [0, 0, 0, 0]

    # Iterate through the predefined list and update results based on the condition
    for i, item in enumerate(predefined_local):
        if item in localisations:
            local_exists[i] = 1  # Update based on presence in localisation

    # Construct the tensor-like structure
    tensor_localisation = [[1, local_exists[0]],
                           [1, local_exists[1]],
                           [1, local_exists[2]],
                           [1, local_exists[3]]]


    for i in tqdm(range(seq_num)):
        # index_f = int(start_time + i*(freq_data/freq_out))
        index_f = start_frame + slide_win * i
        light_color = logs_df.loc[index_f, 'light']
        button_status = logs_df.loc[index_f, 'button']

        node_features = [
            [0.0, 1.0, 0.0],  # Features for Pedestrian node 0
            [1.0, local_exists[0], 0.0],  # Features for SW1 node 1
            [1.0, local_exists[1], 0.0],  # Features for SW2 node 2
            [1.0, local_exists[2], 0.0],  # Features for R node 3
            [1.0, local_exists[3], 0.0],  # Features for H node 4
            [2.0, car_exists[0], 0.0],  # Features for V node 5
            [3.0, 1.0, light_color],  # Features for TL node 6
        ]

        local_t = logs_df.loc[index_f, 'localisation']
        # adjacency list
        node_numbers = {'SW1': 1, 'SW2': 2, 'R': 3, 'H': 4}

        source_nodes = [0]
        target_nodes = [node_numbers[local_t]]

        if car_exists[0] == 1.0:
            source_nodes.append(5)
            target_nodes.append(3)
        if button_status == 1.0:
            source_nodes.append(0)
            target_nodes.append(6)
        if light_color == 1.0:
            source_nodes.append(6)
            target_nodes.append(0)
        elif light_color == 3.0:
            source_nodes.append(6)
            target_nodes.append(5)
        # Edge Index: source -> target
        edge_index = [
            source_nodes,  # Source nodes
            target_nodes  # Target nodes
        ]

        graph_dict = {'node_features': node_features, 'edge_index': edge_index}

        with open(os.path.join(graph_path, f'g_{index_f}.json'), 'w') as json_file:
            json.dump(graph_dict, json_file)





