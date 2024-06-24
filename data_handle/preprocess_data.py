from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_social_media_data(data):
     scaler = StandardScaler()
     scaled_data = scaler.fit_transform(data[['user_id','num_posts', 'num_likes', 'num_comments', 'num_shares']])
     return np.array(scaled_data)

def preprocess_wearable_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['user_id', 'steps_taken', 'hours_of_sleep', 'heart_rate',  'calories_burned']])
    return np.array(scaled_data)

def preprocess_academic_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['user_id', 'GPA', 'attendance', 'num_extracurricular_activities']])
    return np.array(scaled_data)

def preprocess_peer_interaction_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['user_id', 'num_friends', 'messages_sent', 'group_activities_participated']])
    return np.array(scaled_data)