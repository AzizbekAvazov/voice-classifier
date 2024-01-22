import time

import pandas as pd
import torch
import torchaudio
from ml_scripts.cnn import CNNNetwork
from ml_scripts.utils import *


def _resample_if_necessary(signal, sr, device):
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE).to(device)
        signal = resampler(signal)
    return signal


def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def _cut_if_necessary(signal, num_samples):
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal


def _right_pad_if_necessary(signal, num_samples):
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal


def find_unique_classes(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract unique values from the "class" column
    unique_classes = df['class'].unique()

    return unique_classes


def load_model(model_path, device):
    cnn = CNNNetwork().to(device)
    cnn.load_state_dict(torch.load(model_path))
    cnn.eval()
    return cnn


def predict(test_widgets, test_audio_file):
    device = "cpu"

    test_widgets.test_record_btn.configure(state="disabled")
    test_widgets.test_stop_btn.configure(state="disabled")

    class_mapping = find_unique_classes(get_file_path(TRAIN_METADATA_DIR, TRAIN_METADATA_FILENAME))

    model = load_model(get_file_path(MODEL_SAVE_PATH, MODEL_SAVE_NAME), device)

    transformation = mel_spectrogram.to(device)

    signal, sr = torchaudio.load(test_audio_file)
    signal = signal.to(device)
    signal = _resample_if_necessary(signal, sr, device)
    signal = _mix_down_if_necessary(signal)
    signal = _cut_if_necessary(signal, NUM_SAMPLES)
    signal = _right_pad_if_necessary(signal, NUM_SAMPLES)
    signal = transformation(signal)

    signal.unsqueeze_(0)

    with torch.no_grad():
        predictions = model(signal)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]

    test_widgets.test_record_btn.configure(state="normal")
    test_widgets.test_stop_btn.configure(state="normal")

    prediction_label_text = f"Predicted value: {predicted}"

    test_widgets.prediction_label.configure(text=prediction_label_text)
    test_widgets.prediction_label.grid(row=1, column=0, padx=10, pady=(20, 0), columnspan=2, sticky="ew")
