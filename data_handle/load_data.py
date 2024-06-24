import os
import pandas as pd

# Gets the path to the 'data' directory within the project
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_social_media_data(file_name='social_media_data_large.csv'):
    column_names = ['user_id', 'num_posts', 'num_likes', 'num_comments', 'num_shares']
    file_path = os.path.join(data_dir, 'social_media_data_large.csv')
    return pd.read_csv(file_path, names=column_names, header=0)

def load_wearable_data(file_name='wearable_data_large.csv'):
    column_names = ['user_id', 'steps_taken', 'hours_of_sleep', 'heart_rate',  'calories_burned']
    file_path = os.path.join(data_dir, 'wearable_data_large.csv')
    return pd.read_csv(file_path, names=column_names, header=0)

def load_academic_data(file_name='academic_data_large.csv'):
    column_names = ['user_id', 'GPA', 'attendance', 'num_extracurricular_activities']
    file_path = os.path.join(data_dir, 'academic_data_large.csv')
    return pd.read_csv(file_path, names=column_names, header=0)

def load_peer_interaction_data(file_name='peer_interaction_data_large.csv'):
    column_names = ['user_id', 'num_friends', 'messages_sent', 'group_activities_participated']
    file_path = os.path.join(data_dir, 'peer_interaction_data_large.csv')
    return pd.read_csv(file_path, names=column_names, header=0)
