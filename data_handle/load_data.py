import os
import pandas as pd

# Gets the path to the 'data' directory within the project
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')

def load_social_media_data():
    file_path = os.path.join(data_dir, 'social_media_data_large.csv')
    return pd.read_csv(file_path)

def load_wearable_data():
    file_path = os.path.join(data_dir, 'wearable_data_large.csv')
    return pd.read_csv(file_path)

def load_academic_data():
    file_path = os.path.join(data_dir, 'academic_data_large.csv')
    return pd.read_csv(file_path)

def load_peer_interaction_data():
    file_path = os.path.join(data_dir, 'peer_interaction_data_large.csv')
    return pd.read_csv(file_path)
