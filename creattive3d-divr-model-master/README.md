# DiVR: Incorporating Context from Diverse VR Scenes for Human Trajectory Prediction

This repository contains the code accompanying our ECCV CV4Metaverse publication, "DiVR: Incorporating Context from Diverse VR Scenes for Human Trajectory Prediction." Citing our work:

```
@inproceedings{franco2024divr,
  author = {Franz Franco Gallo and Hui-Yin Wu and Lucile Sassatelli},
  title = {DiVR: Incorporating Context from Diverse VR Scenes for Human Trajectory Prediction},
  year = {2024},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  series = {Workshop CV4Metaverse},
}
```

## CREATTIVE3D Dataset for DiVR Model Training

The **CREATTIVE3D** dataset supports the development of models for human trajectory prediction in diverse VR scenarios, such as DiVR. This README provides instructions for accessing and preprocessing the dataset for training.

---

## Dataset Access

The dataset is hosted on Zenodo. You can download it using the link below:

**[CREATTIVE3D Dataset on Zenodo](https://zenodo.org/records/10406560)**

---

## Preprocessing Instructions

Once you have downloaded the dataset, follow these steps to preprocess the data for training the DiVR model. The preprocessing pipeline ensures that the dataset is transformed into the required formats, including motion data, gaze data, and heterogeneous graphs.

### Steps to Preprocess the Dataset

1. **Generate CSV files for training, validation, and testing:**  
   Run the following command to create the split CSV files from the raw dataset:
   ```bash
   python3 ./generate_data/generate_data_train.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train
   ```

2. **Create motion data:**  
   Use the sliding window approach (size: 30) to generate motion data from the CSV file:
   ```bash
   python3 motion_data.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train/ --csv_file dataset_model4.csv --slide_win 30
   ```

3. **Generate gaze data:**  
   Process gaze information from the CSV file using the command below:
   ```bash
   python3 ./generate_data/gaze_pointcloud.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train --csv_file dataset_model4.csv --slide_win 30
   ```

4. **Generate heterogeneous graphs:**  
   Build heterogeneous graphs for the dataset by running the following command:
   ```bash
   python ./generate_data/heterogeneous_graphs.py --data_path ../../GUsT-3D/GustNewFormat/Data --results_path ../GUsT3D_data_train --csv_file dataset_model4.csv --slide_win 30
   ```

---

## Notes
- Ensure all paths provided in the commands align with your local directory structure.
- The preprocessing steps prepare the dataset to be compatible with the DiVR model training pipeline, incorporating motion, gaze, and graph data.

---

## Citation

If you use the CREATTIVE3D dataset or related tools in your research, please cite the corresponding paper and dataset:

**Paper Citation:**
```bibtex
@unpublished{wu:hal-04429351,
  TITLE = {{Exploring, walking, and interacting in virtual reality with simulated low vision: a living contextual dataset}},
  AUTHOR = {Wu, Hui-Yin and Robert, Florent Alain Sauveur and Gallo, Franz Franco and Pirkovets, Kateryna and Quere, Cl{\'e}ment and Delachambre, Johanna and Ramano{\"e}l, Stephen and Gros, Auriane and Winckler, Marco and Sassatelli, Lucile and Hayotte, Meggy and Menin, Aline and Kornprobst, Pierre},
  URL = {https://inria.hal.science/hal-04429351},
  NOTE = {working paper or preprint},
  YEAR = {2023},
  MONTH = Dec,
  KEYWORDS = {Virtual reality ; Dataset ; Context ; Low vision ; 3D environments ; User study},
  PDF = {https://inria.hal.science/hal-04429351v1/file/2023_CREATTIVE3D_dataset_arxiv_.pdf},
  HAL_ID = {hal-04429351},
  HAL_VERSION = {v1},
}
```
# Training DIVR model

## Prerequisites

Before proceeding, ensure you have the following installed on your system:
- Singularity
- Git

## Creating the Singularity Container

### Step 1: Build the Container

Create a writable container using the sandbox option from the Singularity definition file `divr_env.def`:

```bash
sudo singularity build --sandbox divr_env.simg divr_env.def
```

### Step 2: Modify the Container

Open the container with writable permissions:

```bash
sudo singularity shell --writable divr_env.simg/
```

Inside the container, follow these additional setup steps:

#### Vposer - Human Body Prior

```bash
cd /home/human_body_prior
git checkout origin/cvpr19
python3 setup.py develop
```

#### PointNet++

```bash
cd /home/Pointnet2_PyTorch
sed -i '100,101s/^/#/' pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu
sed -i '196,198s/.*/interpolated_feats = known_feats.repeat(1, 1, unknown.shape[1])/' pointnet2_ops_lib/pointnet2_ops/pointnet2_modules.py
pip3 install -r requirements.txt
pip3 install -e .
cp -r pointnet2 /usr/local/lib/python3.8/dist-packages/
```

### Step 3: Build the Singularity Container for Production

After completing all modifications, exit the container:

```bash
exit
```

Build the container for production from the writable sandbox container:

```bash
sudo singularity build divr_env.sif divr_env.simg
```

## Run Evaluation Test on the DiVR Model

### Option 1: Run Evaluation with Singularity Container (Ubuntu Only)

After creating the Singularity container `divr_env.sif`:

```bash
cd $DiVR_FOLDER
bash scripts/test_singularity.sh demo_data $SINGULARITY_FOLDER
```

### Option 2: Run Evaluation with Local Environment

After setting up a local environment with `requirements.txt`:

```bash
cd $DiVR_FOLDER
bash scripts/test_local.sh demo_data 
```


