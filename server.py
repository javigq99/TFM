import tkinter as tk
from tkinter import filedialog
import numpy as np
import socket
import librosa
import pickle

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '192.168.0.205'
port = 12345
sr = 16000
segment_duration = 5
server.bind((host, port))
server.listen(5)

def send_audio_data(audio_path):
    global sr
    y, sr = librosa.load(audio_path, sr=sr)
    audio_duration = librosa.get_duration(y=y, sr=sr)
    mfcc_por_segmento = []

    for inicio in range(0, int(audio_duration), segment_duration):
        end = inicio + segment_duration
        if end > audio_duration:
            break
        
        segmento = y[inicio * sr:end * sr]
        #mfcc = librosa.feature.mfcc(y=segmento, sr=sr).reshape(20, 157, 1)
        #mfcc_por_segmento.append(mfcc)
        mfcc_por_segmento.append(segmento)
        res = np.array(mfcc_por_segmento)
        print(res.shape)
    return res

if __name__ == '__main__':
    client, direccion = server.accept()
    print(f"Connected: {direccion}")
    
    while True:
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Selecciona un archivo",
            filetypes=[
                ("Todos los archivos", "*.*"),
                ("Archivos de audio (WAV, MP3, MP4)", "*.wav *.mp3 *.mp4")
            ]
        )

        if file_path:
            print("File selected:", file_path)
        else:
            print("No file selected")
            client.sendall(b"BYE")
            client.close()
            server.close()
            break

        audio_data = pickle.dumps(send_audio_data(file_path))

        size = len(audio_data)
        client.sendall(size.to_bytes(8, 'big'))
        client.sendall(audio_data)
