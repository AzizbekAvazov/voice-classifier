import csv
import os
import sys

import torchaudio

APP_NAME = "Voice Classifier App"
MAIN_WINDOW_WIDTH_PERCENT = 30
MAIN_WINDOW_HEIGHT_PERCENT = 40

TRAIN_DIR = "ml_scripts/dataset/train"
TRAIN_METADATA_DIR = "ml_scripts/dataset/"
TRAIN_METADATA_FILENAME = "train_metadata.csv"

TEST_DIR = "ml_scripts/dataset/test"

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

MODEL_SAVE_NAME = "feedforwardnet.pth"
MODEL_SAVE_PATH = "ml_scripts/models"

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def get_file_path(dir_path, filename):
    """ Create a file path that works for dev and for PyInstaller """
    abs_path = resource_path(dir_path)
    os.makedirs(abs_path, exist_ok=True)
    return os.path.join(abs_path, filename)


def log_into_csv(data):
    """ Logs into csv newly created audio file for training """
    log_file_path = get_file_path(TRAIN_METADATA_DIR, TRAIN_METADATA_FILENAME)
    with open(log_file_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writerow(data)


def truncate_metadata():
    """
    Deletes all the records from the train_metadata.csv
    """
    # clear the train_metadata.csv
    metadata_file = get_file_path(TRAIN_METADATA_DIR, TRAIN_METADATA_FILENAME)

    if os.path.isfile(metadata_file):
        with open(metadata_file, "r+") as file:
            file.readline()
            file.truncate(file.tell())


def delete_train_dataset():
    """
    Deletes all the recorded audio files
    """
    # delete audio files located in dataset/train
    train_folder = resource_path(TRAIN_DIR)
    try:
        for filename in os.listdir(train_folder):
            os.remove(f"{train_folder}/{filename}")
    except FileNotFoundError:
        pass
