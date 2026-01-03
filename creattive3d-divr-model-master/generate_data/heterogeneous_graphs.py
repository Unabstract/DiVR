# usage: python ./generate_data/heterogeneous_graphs.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train_new --csv_file dataset_model4.csv --slide_win 30

import json
import argparse
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from utils.graph_utils import *
import matplotlib.pyplot as plt

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
data_folder = Path(args.data_path)

# results_path = '../../GUsT3D_data_train_new'
results_folder = Path(args.results_path)
train_path = results_folder / args.csv_file
scenes_path = results_folder / 'scene_nodes.csv'

# reading training file
train_df = pd.read_csv(train_path)

# scene data
# reading node scene bounding boxes
scenes_df = pd.read_csv(scenes_path)

# processing node features: x_min, y_min, x_max, y_max, centr_x, centr_y
scenes_df['y_min'] = scenes_df['z_max'] * -1
scenes_df['y_max'] = scenes_df['y_min'] + scenes_df['scale_z'] * scenes_df['size_z']
scenes_df['x_min'] = scenes_df['x_max'] - scenes_df['scale_x'] * scenes_df['size_x']
scenes_df['x_cent'] = (scenes_df['x_min'] + scenes_df['x_max']) / 2
scenes_df['y_cent'] = (scenes_df['y_min'] + scenes_df['y_max']) / 2

scenes_df = scenes_df[['scene', 'node', 'x_min', 'y_min', 'x_max', 'y_max', 'x_cent', 'y_cent']]

# Normalizing values
x_scene = 4
y_scene = 10
d_scene = (x_scene**2 + y_scene**2)**0.5

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
    logs_df = pd.read_csv(log_file, usecols=['unixTime', 'position', 'localisation', 'light', 'button', 'carPosition'])

    # Convert unixTime
    logs_df['dateTime'] = pd.to_datetime(logs_df['unixTime'], unit='ms')
    logs_df = logs_df.set_index('dateTime')
    # Preliminary cleaning and preparation
    light_mapping = {'red': 0, 'orange': 1, 'green': 2}
    localisation_mapping = {'start_sidewalk': 'SSW','Crossing': 'R', 'Road': 'R', 'opposite_sidewalk': 'ESW',
                            'outside': 'SSW', 'House':'H'}
    # light
    logs_df['light'] = logs_df['light'].map(light_mapping)
    # localisation
    logs_df['localisation'] = logs_df['localisation'].map(localisation_mapping)

    # Determine if the button was pressed by the person
    logs_df['button'] = logs_df['button'].apply(lambda x: 1 if x == 'interact' else 0)

    # Check for vehicle presence and extract coordinates
    logs_df['carPresence'] = logs_df['carPosition'].apply(lambda x: 1 if x != np.NaN else 0)
    logs_df['car_x'] = logs_df['carPosition'].apply(lambda x: safe_eval(x).get('x') if pd.notna(x) else -10)
    logs_df['car_y'] = logs_df['carPosition'].apply(lambda x: -safe_eval(x).get('z') if pd.notna(x) else -10)

    # extract user coordinates
    logs_df['user_x'] = logs_df['position'].apply(lambda x: safe_eval(x).get('x') if pd.notna(x) else -10)
    logs_df['user_y'] = logs_df['position'].apply(lambda x: -safe_eval(x).get('z') if pd.notna(x) else -10)

    # drop unused columns
    logs_df = logs_df.drop(columns=['position', 'carPosition'])

    # fill nan
    logs_df = logs_df.fillna(logs_df.mode().iloc[0])

    # sampling
    logs_df.drop('unixTime', axis=1, inplace=True)
    logs_df = logs_df.resample('16.66ms').ffill()
    logs_df = logs_df.iloc[1:]

    logs_df = logs_df[(logs_df.index >= start_datetime) & (logs_df.index <= end_datetime)]
    logs_df = logs_df.reset_index()
    logs_df.set_index(logs_df.index + row_train['start_frame'], inplace=True)

    graph_path = results_folder / f'{scene}' /f'{seq}'/ 'graph_data_het'
    scene_i = scenes_df.loc[scenes_df['scene'] == scene]

    # nodes
    home = Node(id=0, interactable=0, localization=1, movable=0, color=-1, presence=0,
                x_min=-40, y_min=-100, x_max=-40, y_max=-100, x_cent=-40, y_cent=100)
    start_sw = Node(id=1, interactable=0, localization=1, movable=0, color=-1, presence=0,
                x_min=-40, y_min=-100, x_max=-40, y_max=-100, x_cent=-40, y_cent=100)
    road = Node(id=2, interactable=0, localization=1, movable=0, color=-1, presence=0,
                x_min=-40, y_min=-100, x_max=-40, y_max=-100, x_cent=-40, y_cent=100)
    end_sw = Node(id=3, interactable=0, localization=1, movable=0, color=-1, presence=0,
                x_min=-40, y_min=-100, x_max=-40, y_max=-100, x_cent=-40, y_cent=100)
    button = Node(id=4, interactable=1, localization=0, movable=0, color=-1, presence=0,
                x_min=-40, y_min=-100, x_max=-40, y_max=-100, x_cent=-40, y_cent=100)
    traffic_l = Node(id=5, interactable=0, localization=0, movable=0, color=-1, presence=0,
                x_min=-40, y_min=-100, x_max=-40, y_max=-100, x_cent=-40, y_cent=100)
    user = Node(id=6, interactable=0, localization=0, movable=1, color=-1, presence=0,
                x_min=-40, y_min=-100, x_max=-40, y_max=-100, x_cent=-40, y_cent=100)
    car = Node(id=7, interactable=1, localization=0, movable=1, color=-1, presence=0,
               x_min=-40, y_min=-100, x_max=-40, y_max=-100, x_cent=-40, y_cent=100)

    # Mapping of DataFrame node identifiers to Node instances
    node_mapping = {
        'H': home,
        'SSW': start_sw,
        'R': road,
        'ESW': end_sw
    }

    # Updating node_mapping either with button/traffic light
    # edges for traffic lights(pole) and the button

    e_traffic_l = Edge(node_a=traffic_l.id, node_b=-1, dynamic_type=0, adjacent_type=0, location_type=1,
                       interaction_type=0, active=0, distance=-100)

    e_button = Edge(node_a=button.id, node_b=-1, dynamic_type=0, adjacent_type=0, location_type=1,
                    interaction_type=0, active=0, distance=-100)

    user_start_x = logs_df.iloc[0, logs_df.columns.get_loc('user_x')]
    user_start_y = logs_df.iloc[0, logs_df.columns.get_loc('user_y')]

    if scene in ['SI0V']:
        e_traffic_l.active = 0
        e_button.active = 0
        SSW_x = scene_i.loc[scene_i['node'] == 'SSW', 'x_cent'].iloc[0]
        SSW_y = scene_i.loc[scene_i['node'] == 'SSW', 'y_cent'].iloc[0]
        ESW_x = scene_i.loc[scene_i['node'] == 'ESW', 'x_cent'].iloc[0]
        ESW_y = scene_i.loc[scene_i['node'] == 'ESW', 'y_cent'].iloc[0]
        d_user_SSW = ((user_start_x-SSW_x)**2 + (user_start_y-SSW_y)**2) ** 0.5
        d_user_ESW = ((user_start_x-ESW_x)**2 + (user_start_y-ESW_y)**2) ** 0.5
        if d_user_ESW < d_user_SSW:
            e_traffic_l.node_b = start_sw.id
            e_button.node_b = end_sw.id
        else:
            e_traffic_l.node_b = end_sw.id
            e_button.node_b = start_sw.id


    else:
        pole_SSW_x = scene_i.loc[scene_i['node'] == 'pole_SSW', 'x_cent'].iloc[0]
        pole_SSW_y = scene_i.loc[scene_i['node'] == 'pole_SSW', 'y_cent'].iloc[0]
        pole_ESW_x = scene_i.loc[scene_i['node'] == 'pole_ESW', 'x_cent'].iloc[0]
        pole_ESW_y = scene_i.loc[scene_i['node'] == 'pole_ESW', 'y_cent'].iloc[0]
        d_pole_SSW = ((user_start_x-pole_SSW_x)**2 + (user_start_y-pole_SSW_y)**2) ** 0.5
        d_pole_ESW = ((user_start_x-pole_ESW_x)**2 + (user_start_y-pole_ESW_y)**2) ** 0.5
        if scene in ['SI1V', 'SI2V']:
            e_traffic_l.active = 1
            e_button.active = 0

            if d_pole_ESW < d_pole_SSW:
                pole_mapping = {'pole_SSW': traffic_l}
                e_traffic_l.node_b = start_sw.id
                e_button.node_b = end_sw.id
                e_traffic_l.distance = d_pole_SSW
            else:
                pole_mapping = {'pole_ESW': traffic_l}
                e_traffic_l.node_b = end_sw.id
                e_button.node_b = start_sw.id
                e_traffic_l.distance = d_pole_ESW

        else:
            e_traffic_l.active = 1
            e_button.active = 1
            if d_pole_ESW < d_pole_SSW:
                pole_mapping = {'pole_ESW': button, 'pole_SSW': traffic_l}
                e_traffic_l.node_b = start_sw.id
                e_traffic_l.distance = d_pole_SSW
                e_button.node_b = end_sw.id
                e_button.distance = d_pole_ESW

            else:
                pole_mapping = {'pole_SSW': button, 'pole_ESW': traffic_l}
                e_traffic_l.node_b = end_sw.id
                e_traffic_l.distance = d_pole_ESW
                e_button.node_b = start_sw.id
                e_button.distance = d_pole_SSW

        node_mapping.update(pole_mapping)

    # Iterate through the DataFrame rows and update Node instances
    for index, row in scenes_df.iterrows():
        # Identify the correct Node instance
        node = node_mapping.get(row['node'])
        # Update the Node instance if it exists
        if node:
            node.x_min = row['x_min']
            node.y_min = row['y_min']
            node.x_max = row['x_max']
            node.y_max = row['y_max']
            node.x_cent = row['x_cent']
            node.y_cent = row['y_cent']
            node.presence = 1

    nodes = [home, start_sw, road, end_sw, button, traffic_l, user, car]

    # Filter nodes where 'presence' attribute is 1
    nodes_with_presence = [node for node in nodes if node.presence == 1]

    # scene edges - static
    e_home_ssw = Edge(node_a=home.id, node_b=start_sw.id, dynamic_type=0, adjacent_type=1, location_type=0,
                      interaction_type=0, active=0, distance=-100)
    e_ssw_r = Edge(node_a=start_sw.id, node_b=road.id, dynamic_type=0, adjacent_type=1, location_type=0,
                   interaction_type=0, active=0, distance=-100)
    e_r_esw = Edge(node_a=road.id, node_b=end_sw.id, dynamic_type=0, adjacent_type=1, location_type=0,
                   interaction_type=0, active=0, distance=-100)

    if home.presence==1 and start_sw.presence==1:
        e_home_ssw.presence = 1
        e_home_ssw.distance = ((home.x_cent- start_sw.x_cent)**2 + (home.y_cent- start_sw.y_cent)**2) ** 0.5
    if start_sw.presence==1 and road.presence==1:
        e_ssw_r.presence = 1
        e_ssw_r.distance = ((start_sw.x_cent - road.x_cent) ** 2 + (start_sw.y_cent - road.y_cent) ** 2) ** 0.5
    if road.presence==1 and end_sw.presence==1:
        e_r_esw.presence = 1
        e_r_esw.distance = ((road.x_cent - end_sw.x_cent) ** 2 + (road.y_cent - end_sw.y_cent) ** 2) ** 0.5

    # dynamic edges
    e_user = Edge(node_a=user.id, node_b=-1, dynamic_type=1, adjacent_type=0, location_type=1,
                  interaction_type=0, active=0, distance=-100)
    e_car = Edge(node_a=car.id, node_b=road.id, dynamic_type=1, adjacent_type=0, location_type=1,
                 interaction_type=0, active=0, distance=-100)
    e_u_button = Edge(node_a=user.id, node_b=button.id, dynamic_type=1, adjacent_type=0, location_type=0,
                      interaction_type=1, active=0, distance=-100)
    e_tl_user = Edge(node_a=traffic_l.id, node_b=user.id, dynamic_type=1, adjacent_type=0, location_type=0,
                     interaction_type=1, active=0, distance=-100)
    e_tl_car = Edge(node_a=traffic_l.id, node_b=car.id, dynamic_type=1, adjacent_type=0, location_type=0,
                    interaction_type=1, active=0, distance=-100)

    if not graph_path.exists():
        Path.mkdir(graph_path, parents=True)

    slide_win = args.slide_win
    seq_num = (end_frame - start_frame) // slide_win  # +1


    # features: x_min, y_min, x_max, y_max, centr_x, centr_y, interactable, localization, movable, color, presence
    for i in tqdm(range(seq_num)):
        # index_f = int(start_time + i*(freq_data/freq_out))
        index_f = start_frame + slide_win * i
        button_status = logs_df.loc[index_f, 'button']

        traffic_l.color = logs_df.loc[index_f, 'light']

        user.x_cent = logs_df.loc[index_f, 'user_x']
        user.y_cent = logs_df.loc[index_f, 'user_y']
        user.presence = 1
        car.x_cent = logs_df.loc[index_f, 'car_x']
        car.y_cent = logs_df.loc[index_f, 'car_y']
        if car.x_cent != -10:
            car.presence = 1

        # Define a list of nodes for the example
        nodes = [home, start_sw, road, end_sw, button, traffic_l, user, car]

        # Convert all nodes to their feature representation
        node_features = [node_to_features(node, x_norm=x_scene, y_norm=y_scene) for node in nodes]

        # edges
        e_user.node_b = node_mapping[logs_df.loc[index_f, 'localisation']].id
        e_user.distance = ((user.x_cent - nodes[e_user.node_b].x_cent) ** 2 + (user.y_cent - nodes[e_user.node_b].y_cent) ** 2) ** 0.5

        if car.presence == 1:
            e_car.active = 1
            e_car.distance = ((car.x_cent-road.x_cent)**2 + (car.y_cent-road.y_cent)**2)**0.5

        if button_status == 1.0:
            e_u_button.active = 1
        if button.x_cent != -40:
            e_u_button.distance = ((button.x_cent-user.x_cent)**2 + (button.y_cent-user.y_cent)**2)**0.5

        if traffic_l.color == 0.0:
            e_tl_user.active = 1
        if traffic_l.x_cent != -40:
            e_tl_user.distance = ((traffic_l.x_cent - user.x_cent) ** 2 + (traffic_l.y_cent - user.y_cent) ** 2) ** 0.5

        if traffic_l.color == 2.0:
            e_tl_car.active = 1

        if car.x_cent != -40 and traffic_l.x_cent != -40:
            e_tl_car.distance = ((traffic_l.x_cent - car.x_cent) ** 2 + (traffic_l.y_cent - car.y_cent) ** 2) ** 0.5

        # defining the list of edges
        edges = [e_home_ssw, e_ssw_r, e_r_esw, e_traffic_l, e_button, e_user, e_car, e_u_button, e_tl_user, e_tl_car]

        source_nodes = []
        target_nodes = []
        for edge in edges:
            source_node, target_node = edge_to_index(edge)
            source_nodes.append(source_node)
            target_nodes.append(target_node)

        # Edge Index: source -> target
        edge_index = [
            source_nodes,  # Source nodes
            target_nodes  # Target nodes
        ]

        edge_features = [edge_to_features(edge, dist_norm=d_scene) for edge in edges]

        graph_dict = {'node_features': node_features, 'edge_features':edge_features, 'edge_index': edge_index}

        with open(os.path.join(graph_path, f'g_{index_f}.json'), 'w') as json_file:
            json.dump(graph_dict, json_file)
