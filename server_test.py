import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import confusion_matrix, classification_report
import socket
import random
from tqdm import tqdm
import os
import pickle


diseases = {0: "Bronchiectasis", 1: "Bronchiolitis", 2: "COPD", 3: "Healthy", 4: "Pneumonia", 5: "URTI"}
diseases_reverse = {v:k for k,v in diseases.items()}

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data')

def split_audio(audio_file, segment_duration=5, sr_new=16000, overlapping=0.3):
    x, sr = librosa.load(audio_file, sr=sr_new)
    
    segment_length = segment_duration * sr_new
    
    if overlapping == 0:
        step_size = segment_length
    else:
        step_size = int(segment_length * (1 - overlapping))
    
    total_length = x.shape[0]
    segments = []
    
    for start in range(0, total_length - segment_length + 1, step_size):
        segment = x[start:start + segment_length]
        segments.append(segment)

    if start + step_size < total_length:
        last_segment = x[-segment_length:]
        last_segment = np.pad(last_segment, (0, max(0, segment_length - last_segment.shape[0])))
        segments.append(last_segment)
    
    return segments

def load_test():
    x = [] 
    y = []
    for label in tqdm(os.listdir(data_path), desc=""):
        folder_path = os.path.join(data_path, label)
        file_list = os.listdir(folder_path)
        if label != 'Asthma' and label != 'LRTI':
            for f in file_list:
                file_path = os.path.join(folder_path, f)
                aux = split_audio(file_path,  overlapping = 0.5)
                for a in aux:
                    x.append(np.array(a))
                    y.append(diseases_reverse[label])
    return np.array(x), np.array(y)

def write_log(y_true, y_pred, file_path="logtransformer.txt"):
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    
    log = f"Number of samples: {len(y_true)}\n"
    log += f"Accuracy: {accuracy:.4f}\n\n"
    log += f"Confusion Matrix:\n{conf_matrix}\n\n"
    log += f"Classification Report:\n{class_report}\n"
    
    with open(file_path, "w") as f:
        f.write(log)

if __name__ == '__main__':

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_name = socket.gethostname()
    host = socket.gethostbyname_ex(host_name)[2][2]
    X,Y = load_test()
    port = 12345
    sr = 16000
    segment_duration = 5
    server.bind((host, port))
    server.listen(5)
    
    client, direccion = server.accept()
    print(f"Connected: {direccion}")

    Y_pred = []
    for data in tqdm(X, desc=""):
        audio_data = pickle.dumps(data)
        size = len(audio_data)
        client.sendall(size.to_bytes(8, 'big'))
        client.sendall(audio_data)

        ack = client.recv(10).decode('utf-8')
        ack = ack.split(",")

        if len(ack) == 2 and ack[0] == "ACK":
            Y_pred.append(int(ack[1]))

    client.sendall(b"BYE")
    client.close()
    server.close()
    write_log(Y, Y_pred)


