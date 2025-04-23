import socket
import pickle
import numpy as np
import sys
from utils.mfccs import compute_mfccs
from neural_network_class import NeuralNetwork



if __name__ == '__main__':
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    host = sys.argv[1]
    model = sys.argv[2]
    puerto = 12345
    model = NeuralNetwork("models/" + model)
    client.connect((host, puerto))

    while True:

        tamaño_datos = client.recv(8)
        if not tamaño_datos:
            break
        
        tamaño = int.from_bytes(tamaño_datos, 'big')

        if tamaño_datos == b'BYE':
            print("Connection closed by the server")
            client.close()
            break

        data = b""
        while len(data) < tamaño:
            paquete = client.recv(4096)
            if not paquete:
                break
            data += paquete

        if data:
            audio_data = pickle.loads(data)
            mfccs = compute_mfccs(audio_data, sample_rate=16000, n_mfcc=20, n_fft=2048, hop_length=512, window='hann',num_filters=128,htk=False)
            model.launch_inference(mfccs)
            classification = model.get_results()
            disease = np.argmax(classification)
            ack_msg = f"ACK,{disease}".encode('utf-8')
            client.sendall(ack_msg)
