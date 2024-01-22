import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from ml_scripts.AppDataset import AppDataset
from ml_scripts.cnn import CNNNetwork
from ml_scripts.utils import *

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = get_file_path(TRAIN_METADATA_DIR, TRAIN_METADATA_FILENAME)
AUDIO_DIR = resource_path(TRAIN_DIR)


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs, progress_bar):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        progress_bar.step()
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")

    progress_bar.stop()
    progress_bar.grid_forget()
    print("Finished training")


def start_training(*args):
    progress_bar, train_widgets = args[0], args[1]

    progress_bar.grid(row=3, column=0, padx=10, pady=(20, 0), columnspan=2, sticky="ew")
    progress_bar.start()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")

    usd = AppDataset(ANNOTATIONS_FILE,
                     AUDIO_DIR,
                     mel_spectrogram,
                     SAMPLE_RATE,
                     NUM_SAMPLES,
                     device)

    train_dataloader = create_data_loader(usd, BATCH_SIZE)
    cnn = CNNNetwork().to(device)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    train_widgets.train_record_btn.configure(state="disabled")
    train_widgets.train_stop_btn.configure(state="disabled")

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS, progress_bar)

    model_path = get_file_path(MODEL_SAVE_PATH, MODEL_SAVE_NAME)

    # save model
    torch.save(cnn.state_dict(), model_path)
    print("Trained feed forward net saved at feedforwardnet.pth")

    train_widgets.train_record_btn.configure(state="normal")
    train_widgets.train_stop_btn.configure(state="normal")
