import os
import glob
from typing import List
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# def get_csv_files(root_location: str, subject_ids: List[str], keywords=['i0','i5','i10','d5','d10','SA','SD'], shuffle=True) -> List[str]:
# def get_csv_files(root_location: str, subject_ids: List[str], keywords=['walk','SD','SA'], shuffle=True) -> List[str]:
def get_csv_files(root_location: str, subject_ids: List[str], keywords=['SA'], shuffle=True) -> List[str]:
    search_pattern = os.path.join(root_location, '**', '*.csv')
    files = glob.glob(search_pattern, recursive=True)
    csv_files = []
    for f in files:
        basename = os.path.basename(f)
        if any(f'_{subject_id}_' in basename for subject_id in subject_ids):
            if any(f'_{kw}_' in basename for kw in keywords):
                csv_files.append(f)
    if shuffle:
        random.shuffle(csv_files)
    return csv_files


def load_data_file(csv_files, winlen=200,feature_columns=['left_hip_angle', 'left_knee_angle', 'right_hip_angle', 'right_knee_angle'],frame_per_session=None):
    """
    Load the dataset and create sliding windows of data and labels.
    """
    # data_set = []
    # label_set = []
    data_set, gait_label_set, incline_label_set = [], [], []

    for csv_idx, file in enumerate(csv_files):
        df = pd.read_csv(file)
        features = df[feature_columns].values
        gait_labels = df['gait_phase'].values - 1  # Adjust labels to start from 0
        incline_labels = df['terrain_info'].values  # Add incline angle

        if not frame_per_session:
            num_windows = len(gait_labels) - winlen
        else:
            num_windows = frame_per_session
        if num_windows <= 0:
            continue  # Skip if data is insufficient

        # trial_features = []
        # trial_labels = []
        trial_features, trial_gait_labels, trial_incline_labels = [], [], []
        for idx in range(num_windows):
            feature_window = features[idx:idx + winlen]
            gait_label_window = gait_labels[idx + winlen - 1]  # Label at the end of the window
            incline_labels_window = incline_labels[idx + winlen - 1]
            trial_features.append(feature_window)
            trial_gait_labels.append(gait_label_window)
            trial_incline_labels.append(incline_labels_window)

        # Store all windowed data for the current trial
        data_set.append(np.array(trial_features))
        gait_label_set.append(np.array(trial_gait_labels))
        incline_label_set.append(np.array(trial_incline_labels))

    return data_set, gait_label_set, incline_label_set

def load_data_file_lgi(csv_files, winlen,feature_columns=['left_hip_angle', 'left_knee_angle', 'right_hip_angle', 'right_knee_angle'],max_samples=None):
        """
        Load the dataset and create sliding windows of data and labels.
        """
        # data_set = []
        # label_set = []
        data_set, gait_label_set, incline_label_set, loc_label_set = [], [], [], []

        for csv_idx, file in enumerate(csv_files):
            df = pd.read_csv(file)
            # Limit to the first `max_samples` rows
            df = df.iloc[:max_samples]

            features = df[feature_columns].values
            gait_labels = df['gait_phase'].values-1  # Adjust labels to start from 0
            incline_labels = df['terrain_info'].values  # Add incline angle

            # Mapping dictionary
            label_mapping = {
                100: 0,  # Flat ground walking
                201: 1,  # Incline walking
                202: 2,  # Decline walking
                301: 3,  # Stair ascending
                302: 4  # Stair descending
            }

            # Apply mapping to labels
            df['locomotion_mode'] = df['locomotion_mode'].map(label_mapping) # Add locomotion mode
            locomotion_labels = df['locomotion_mode'].values

            # Create Sliding Windows
            num_windows = len(gait_labels) - winlen
            if num_windows <= 0:
                continue  # Skip if data is insufficient

            # trial_features = []
            # trial_labels = []
            trial_features, trial_gait_labels, trial_incline_labels, trial_loc_labels = [], [], [], []
            for idx in range(num_windows):
                feature_window = features[idx:idx + winlen]
                gait_label_window = gait_labels[idx + winlen - 1]  # Label at the end of the window
                incline_labels_window = incline_labels[idx + winlen - 1]
                loc_labels_window = locomotion_labels[idx + winlen - 1]
                trial_features.append(feature_window)
                trial_gait_labels.append(gait_label_window)
                trial_incline_labels.append(incline_labels_window)
                trial_loc_labels.append(loc_labels_window)

            # Store all windowed data for the current trial
            data_set.append(np.array(trial_features))
            gait_label_set.append(np.array(trial_gait_labels))
            incline_label_set.append(np.array(trial_incline_labels))
            loc_label_set.append(np.array(trial_loc_labels))

        return data_set, gait_label_set, incline_label_set, loc_label_set


# DataLoader Function
def prepare_dataloader(data, g_label, i_gt, l_label, device, batch_size):
    data = torch.tensor(np.concatenate(data, axis=0)).permute(0, 2, 1).float()
    g_label = torch.tensor(np.concatenate(g_label, axis=0), dtype=torch.long)
    i_gt = torch.from_numpy(np.concatenate(i_gt, axis=0)).float()
    l_label = torch.tensor(np.concatenate(l_label, axis=0), dtype=torch.long)
    
    total_before = data.shape[0]
    valid_gait = (g_label >= 0) & (g_label <= 3)
    valid_loco = (l_label >= 0) & (l_label <= 4)
    valid_incline = ~torch.isnan(i_gt)
    valid_indices = valid_gait & valid_loco & valid_incline

    # Apply filtering
    data = data[valid_indices]
    g_label = g_label[valid_indices]
    i_gt = i_gt[valid_indices]
    l_label = l_label[valid_indices]
    total_after = data.shape[0]

    data, g_label, i_gt, l_label = data.to(device), g_label.to(device), i_gt.to(device), l_label.to(device)    # Move to device

    dataset = TensorDataset(data, g_label, i_gt, l_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # print(f"Data Loader: Total data points before filtering: {total_before}")
    # print(f"Data Loader: Total data points after filtering: {total_after}")
    #print(f"Unique gait labels: {torch.unique(g_label)}")
    #print(f"Unique incline labels: {torch.unique(i_gt)}")
    #print(f"Unique loco labels: {torch.unique(l_label)}")
    #print('Dataloader finished')
    
    return dataloader