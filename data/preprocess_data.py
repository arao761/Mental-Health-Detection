from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_social_media_data(data):
    return data['text'].tolist()

def preprocess_wearable_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop(columns=['timestamp']))
    return np.array(scaled_data)

def preprocess_academic_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop(columns=['student_id']))
    return np.array(scaled_data)

def preprocess_peer_interaction_data(data):
    return data.drop(columns=['interaction_id']).values
