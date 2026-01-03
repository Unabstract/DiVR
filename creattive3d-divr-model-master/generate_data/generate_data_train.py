# usage: python3 ./generate_data/generate_data_train.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train_new

import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',
                    type=str,
                    required=True,
                    help='Path to the GUsT3D data folder'
                    )

parser.add_argument('--results_path',
                    type=str,
                    required=True,
                    help='Path to the folder in which we are going to generate the data for training')

args = parser.parse_args()
data_folder = Path(args.data_path)

quest_path = data_folder / 'questionnaire.csv'
condition_list = ['RWNV', 'RWLV']
scene_list = ['CI0V', 'CI1V', 'CI2V', 'SI0V', 'SI1V', 'SI2V']

# list with sequences data to drop due to collection issues
drop_list = ['U899_RWNV_SI1V', 'U940_RWLV_CI1V', 'U091_RWLV_CI1V', 'U511_RWNV_CI0V']

# Training/validation percentage
n_train = 0.8
quest_df = pd.read_csv(quest_path, sep=';', encoding='latin-1')


print('###################################################')
print(f'### Building training dataset ####')
print('###################################################')

df_task_times = pd.DataFrame([])
df_task_training = pd.DataFrame([])
dataset = pd.DataFrame([])
user_list = quest_df.loc[:39, 'uid']

for subject in tqdm(user_list):
    for condition in condition_list:
        df_parsed = pd.DataFrame([])
        for scene in scene_list:
            notation = f'U{int(subject):03d}_{condition}_{scene}'

            if notation in drop_list:
                continue

            gust_file = data_folder / f'U{int(subject):03d}'/ f'{condition}'/ f'{scene}'/f'{notation}_LogP.json'
            pos_file = data_folder / f'U{int(subject):03d}' / f'{condition}' / f'{scene}' / f'{notation}_motion_pos.csv'
            try:
                with open(gust_file, 'r') as json_file:
                    json_load = json.load(json_file)
            except FileNotFoundError:
                print(f'File missing: {notation}_LogP.json')
                continue

            df_pos = pd.read_csv(pos_file)

            data=json_load['logs']
            df = pd.DataFrame(data)
            df1 = df[df['currentTask'].str.match('Step')]

            df1[['step', 'task']] = df1.currentTask.str.split("\n", expand=True)
            df1['sequence_path'] = notation
            df1['scene'] = f'{scene}'

            df_init_time = df1.iloc[:1]
            df_tasks = df1.drop_duplicates(subset=['currentTask'], keep='last')

            # saving data with all columns
            df2 = pd.concat([df_init_time, df_tasks])
            df_task_times = pd.concat([df_task_times, df2])

            # creating more process data for task times
            col_process = ['sequence_path', 'unixTime', 'scene', 'task']
            df_init_time = df_init_time.loc[:, col_process]
            df_init_time = df_init_time.reset_index()
            df_tasks = df_tasks.loc[:, col_process]

            df_task_short = df2.loc[:, col_process]
            df_tasks['time'] = df_task_short.loc[:, 'unixTime'].diff()[1:] / 1000
            end_frame_values = df_tasks['unixTime'] - df_init_time.loc[0,'unixTime']

            df_tasks['start_frame'] = [0]+list(end_frame_values[:-1])
            df_tasks['end_frame'] = end_frame_values
            df_tasks['start_frame'] = df_tasks['start_frame'] * (60/1000)
            df_tasks['end_frame'] = df_tasks['end_frame'] * (60 / 1000)
            df_tasks['start_frame'] = df_tasks['start_frame'].astype('int')
            df_tasks['end_frame'] = df_tasks['end_frame'].astype('int')
            df_tasks['total_frames'] = df_tasks['end_frame'] - df_tasks['start_frame']
            df_tasks['flag'] = np.where(df_tasks['total_frames'] > 0, 1, 0) # 450 originally
            if scene in ['SI0V', 'SI1V', 'SI2V']:
                df_tasks['task'][df_tasks['task'].str.contains('Go to')] = ['Simple_' + i for i in df_tasks['task']
                                                                            if ('Go to' in i)]
            else:
                df_tasks['task'][df_tasks['task'].str.contains('Go to')] = ['Complex_' + i for i in df_tasks['task'] if ('Go to' in i)]
            # TODO: sometimes the gust time exceeds the frames in csv an get an error,
            #  so we better make sure is not bigger than that
            df_tasks = df_tasks[df_tasks['start_frame'] < df_pos.shape[0]]
            if df_tasks['end_frame'].iloc[-1] > df_pos.shape[0]:
                df_tasks['end_frame'].iloc[-1] = df_pos.shape[0]

            df_task_training = pd.concat([df_task_training, df_tasks])

# Show some stats
# df_task_training[df_task_training['task'].str.contains('Simple_Go to : start_sidewalk')]
# df_task_training.groupby('task')['total_frames'].describe()
# df_task_training[df_task_training['sequence_path'].str.contains('U851_RWLV_CI2V')]

# Is not updating these values ... I'm creating a dataframe for each task

# Complex_Go to : opposite_sidewalk:
df_task_complex_1 = df_task_training[df_task_training['task'].str.contains('Complex_Go to : opposite_sidewalk')]
df_task_complex_1['start_frame'] += 30
end_change_1 = 60
df_task_complex_1['end_frame'] += end_change_1
df_task_complex_1['unixTime'] += end_change_1 * 1000 / 60
df_task_complex_1['total_frames'] = df_task_complex_1['end_frame'] - df_task_complex_1['start_frame']
df_task_complex_1['time'] = df_task_complex_1['total_frames'] / 60

# Complex_Go to : House:
#TODO: we could add for the one total_frames < 400 to be simple - no interaction
df_task_complex_2 = df_task_training[df_task_training['task'].str.contains('Complex_Go to : House')]
df_task_complex_2['start_frame'] += 120
df_task_complex_2.loc[df_task_complex_2['total_frames'] > 1500, 'start_frame'] = \
    df_task_complex_2.loc[df_task_complex_2['total_frames'] > 1500, 'end_frame'] - 1200
df_task_complex_2['total_frames'] = df_task_complex_2['end_frame'] - df_task_complex_2['start_frame']
df_task_complex_2['time'] = df_task_complex_2['total_frames'] / 60

# Complex_Go to : start_sidewalk:
df_task_complex_3 = df_task_training[df_task_training['task'].str.contains('Complex_Go to : start_sidewalk')]
df_task_complex_3['start_frame'] += 120
df_task_complex_3.loc[df_task_complex_3['total_frames'] > 1500, 'start_frame'] = \
    df_task_complex_3.loc[df_task_complex_3['total_frames'] > 1500, 'end_frame'] - 1200
df_task_complex_3['total_frames'] = df_task_complex_3['end_frame'] - df_task_complex_3['start_frame']
df_task_complex_3['time'] = df_task_complex_3['total_frames'] / 60

# Simple_Go to : opposite_sidewalk:
df_task_simple_1 = df_task_training[df_task_training['task'].str.contains('Simple_Go to : opposite_sidewalk')]
# df_task_simple_1['start_frame'] -= 60
end_change_2 = 60
df_task_simple_1['end_frame'] += end_change_2
df_task_simple_1['unixTime'] += end_change_2 * 1000 / 60
df_task_simple_1.loc[df_task_simple_1['total_frames'] > 800, 'start_frame'] = \
    df_task_simple_1.loc[df_task_simple_1['total_frames'] > 800, 'end_frame'] - 600
df_task_simple_1['total_frames'] = df_task_simple_1['end_frame'] - df_task_simple_1['start_frame']
df_task_simple_1['time'] = df_task_simple_1['total_frames'] / 60

#Simple_Go to : House:
df_task_simple_2 = df_task_training[df_task_training['task'].str.contains('Simple_Go to : House')]
df_task_simple_2['start_frame'] += 90
df_task_simple_2.loc[df_task_simple_2['total_frames'] > 1000, 'start_frame'] = \
    df_task_simple_2.loc[df_task_simple_2['total_frames'] > 1000, 'end_frame'] - 600
df_task_simple_2['total_frames'] = df_task_simple_2['end_frame'] - df_task_simple_2['start_frame']
df_task_simple_2['time'] = df_task_simple_2['total_frames'] / 60

# Simple_Go to : start_sidewalk:
df_task_simple_3 = df_task_training[df_task_training['task'].str.contains('Simple_Go to : start_sidewalk')]
df_task_simple_3['start_frame'] += 60

df_task_simple_3.loc[df_task_simple_3['total_frames'] > 1200, 'end_frame'] = \
    df_task_simple_3.loc[df_task_simple_3['total_frames'] > 1200, 'start_frame'] + 1000

df_task_simple_3.loc[df_task_simple_3['total_frames'] > 1200, 'unixTime'] += \
    -(df_task_simple_3.loc[df_task_simple_3['total_frames'] > 1200, 'total_frames'] - 1000) * 1000 / 60

df_task_simple_3['total_frames'] = df_task_simple_3['end_frame'] - df_task_simple_3['start_frame']
df_task_simple_3['time'] = df_task_simple_3['total_frames'] / 60

# Get : black_box
df_task_target_1 = df_task_training[df_task_training['task'].str.contains('Get : black_box')]
df_task_target_1['start_frame'] = df_task_target_1['end_frame'] - 700
df_task_target_1.loc[df_task_target_1['start_frame'] < 0, 'start_frame'] = 0
df_task_target_1['total_frames'] = df_task_target_1['end_frame'] - df_task_target_1['start_frame']
df_task_target_1['time'] = df_task_target_1['total_frames'] / 60

# Place black_box on white_table
df_task_target_2 = df_task_training[df_task_training['task'].str.contains('Place black_box on white_table')]
df_task_target_2['start_frame'] = df_task_target_2['end_frame'] - 600
df_task_target_2.loc[df_task_target_2['start_frame'] < 0, 'start_frame'] = 0
df_task_target_2['total_frames'] = df_task_target_2['end_frame'] - df_task_target_2['start_frame']
df_task_target_2['time'] = df_task_target_2['total_frames'] / 60

# Place garbage_bag in trashcan
df_task_target_3 = df_task_training[df_task_training['task'].str.contains('Place garbage_bag in trashcan')]
df_task_target_3['start_frame'] = df_task_target_3['end_frame'] - 500
df_task_target_3.loc[df_task_target_3['start_frame'] < 0, 'start_frame'] = 0
df_task_target_3['total_frames'] = df_task_target_3['end_frame'] - df_task_target_3['start_frame']
df_task_target_3['time'] = df_task_target_3['total_frames'] / 60

# Place box in trashcan
df_task_target_4 = df_task_training[df_task_training['task'].str.contains('Place box in trashcan')]
df_task_target_4['start_frame'] = df_task_target_4['end_frame'] - 500
df_task_target_4.loc[df_task_target_4['start_frame'] < 0, 'start_frame'] = 0
df_task_target_4['total_frames'] = df_task_target_4['end_frame'] - df_task_target_4['start_frame']
df_task_target_4['time'] = df_task_target_4['total_frames'] / 60

# all tasks
df_all_seq = pd.concat([df_task_complex_1, df_task_complex_2, df_task_complex_3,
                        df_task_simple_1, df_task_simple_2, df_task_simple_3,
                        df_task_target_1, df_task_target_2, df_task_target_3, df_task_target_4])

df_all_seq = df_all_seq[df_all_seq['total_frames'] > 450]

df_all_seq['transformation'] = 'transform_info.json'
df_all_seq['training'] = 0
# for train, val and test sets
random_state = 42

# Getting datasets by complexity tasks:
# NV + simple 392
df_NV_simple = df_all_seq[df_all_seq['sequence_path'].str.contains('NV') &
                          df_all_seq['task'].str.contains('Simple|black_box')]
df_NV_simple = df_NV_simple.reset_index()

df_NV_simple_test = df_NV_simple.sample(frac=0.2, random_state=random_state)
df_NV_simple_train = df_NV_simple.loc[~df_NV_simple.index.isin(df_NV_simple_test.index)]
df_NV_simple_train.loc[df_NV_simple_train.sample(frac=0.8, random_state=random_state).index, 'training'] = 1

# dataset model 1: NV + Simple
dataset_model1 = df_NV_simple_train
test_set1 = df_NV_simple_test

# NV + complex 218
df_NV_complex = df_all_seq[df_all_seq['sequence_path'].str.contains('NV') &
                           (df_all_seq['task'].str.contains('Complex'))]
df_NV_complex = df_NV_complex.reset_index()

df_NV_complex_test = df_NV_complex.sample(frac=0.2, random_state=random_state)
df_NV_complex_train = df_NV_complex.loc[~df_NV_complex.index.isin(df_NV_complex_test.index)]
df_NV_complex_train.loc[df_NV_complex_train.sample(frac=0.8, random_state=random_state).index, 'training'] = 1

# dataset model 2:NV + Simple + Complex
df_NV_simple_train_model2 = df_NV_simple_train.sample(n = df_NV_complex_train.shape[0], random_state=random_state)
df_NV_simple_train_model2['training'] = 0
df_NV_simple_train_model2.loc[df_NV_simple_train_model2.sample(frac=0.8, random_state=random_state).index, 'training'] = 1

dataset_model2 = pd.concat([df_NV_simple_train_model2, df_NV_complex_train])
test_set2 = df_NV_complex_test


# LV + simple 219
df_LV_simple = df_all_seq[df_all_seq['sequence_path'].str.contains('LV') &
                          (df_all_seq['task'].str.contains('Simple'))]
df_LV_simple = df_LV_simple.reset_index()

df_LV_simple_test = df_LV_simple.sample(frac=0.2, random_state=random_state)
df_LV_simple_train = df_LV_simple.loc[~df_LV_simple.index.isin(df_LV_simple_test.index)]
df_LV_simple_train.loc[df_LV_simple_train.sample(frac=0.8, random_state=random_state).index, 'training'] = 1

# dataset model 3: NV + Simple + LV
df_LV_simple_train_model3 = df_NV_simple_train.sample(n = df_LV_simple_train.shape[0], random_state=random_state)
df_LV_simple_train_model3['training'] = 0
df_LV_simple_train_model3.loc[df_LV_simple_train_model3.sample(frac=0.8, random_state=random_state).index, 'training'] = 1

dataset_model3 = pd.concat([df_LV_simple_train_model3, df_LV_simple_train])
test_set3 = df_LV_simple_test

# LV + complex 216
df_LV_complex = df_all_seq[df_all_seq['sequence_path'].str.contains('LV') &
                           (df_all_seq['task'].str.contains('Complex'))]
df_LV_complex = df_LV_complex.reset_index()

df_LV_complex_test = df_LV_complex.sample(frac=0.2, random_state=random_state)
df_LV_complex_train = df_LV_complex.loc[~df_LV_complex.index.isin(df_LV_complex_test.index)]
df_LV_complex_train.loc[df_LV_complex_train.sample(frac=0.8, random_state=random_state).index, 'training'] = 1

# dataset model 4:
df_NV_simple_train_model4 = df_NV_simple_train_model2
dataset_model4 = pd.concat([df_NV_simple_train_model4, df_NV_complex_train, df_LV_complex_train])
test_set4 = df_LV_complex_test

df_all_train = pd.concat([df_NV_simple, df_NV_complex, df_LV_simple, df_LV_complex])

results_path = Path(args.results_path)
if not results_path.exists():
    Path.mkdir(results_path, parents=True)

df_all_seq.to_csv(results_path / f'dataset_all_seq.csv', index=False)
df_all_train.to_csv(results_path / f'dataset_all_train.csv', index=False)
df_LV_simple.to_csv(results_path / f'dataset_LV_simple.csv', index=False)
dataset_model1.to_csv(results_path / f'dataset_model1.csv', index=False)
dataset_model2.to_csv(results_path / f'dataset_model2.csv', index=False)
dataset_model3.to_csv(results_path / f'dataset_model3.csv', index=False)
dataset_model4.to_csv(results_path / f'dataset_model4.csv', index=False)
test_set1.to_csv(results_path / f'test_set1.csv', index=False)
test_set2.to_csv(results_path / f'test_set2.csv', index=False)
test_set3.to_csv(results_path / f'test_set3.csv', index=False)
test_set4.to_csv(results_path / f'test_set4.csv', index=False)
