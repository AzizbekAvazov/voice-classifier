import wave
import customtkinter
import tkinter as tk
import threading
import time
import pandas as pd
import pyaudio
from ml_scripts.utils import *
from augmentation import augment_audio
from ml_scripts.train import start_training


class TrainModelWidgets:
    def __init__(self, parent_frame):
        """
        Initialize the TrainModelWidgets class.

        Parameters:
        - parent_frame (tk.Frame): Parent frame for the widgets.
        """
        self.parent_frame = parent_frame
        self.recording = False

        # Initialize csv metadata file for training dataset if it does not exist
        self.setup_csv_file()

        self.create_widgets()
        self.validate_name_entry()

        # Audio recording variables
        self.frames = []
        self.audio_stream = None
        self.audio_thread = None

    def create_widgets(self):
        """
        Create widgets for the "Train Model" button.
        Initially, they are hidden.
        """
        # Variable to track changes in name entry
        self.train_input_entry_var = tk.StringVar()
        self.train_input_entry_var.trace_add("write", self.validate_name_entry)

        self.train_input_label = customtkinter.CTkLabel(self.parent_frame,
                                                        text="Enter your name:",
                                                        font=("Times", 18, "bold"))

        self.train_input_entry = customtkinter.CTkEntry(self.parent_frame,
                                                        width=300,
                                                        textvariable=self.train_input_entry_var)

        self.train_record_btn = customtkinter.CTkButton(self.parent_frame,
                                                        text="Start recording",
                                                        fg_color="#006400",
                                                        hover_color="#008000")

        self.train_stop_btn = customtkinter.CTkButton(self.parent_frame,
                                                      text="Stop recording",
                                                      fg_color="#343a40",
                                                      hover_color="#495057")

        self.progress_bar = customtkinter.CTkProgressBar(self.parent_frame,
                                                         width=200,
                                                         mode="indeterminate")

        self.training_progress_bar = customtkinter.CTkProgressBar(self.parent_frame,
                                                         width=200,
                                                         mode="determinate")

        self.hide_widgets()

        self.train_record_btn.configure(command=self.start_recording)
        self.train_stop_btn.configure(command=self.stop_recording)

    def validate_name_entry(self, *args):
        """
        Enable or disable the record and stop buttons based on the name entry content.
        """
        name_content = self.train_input_entry_var.get()
        is_name_valid = bool(name_content.strip())

        if is_name_valid:
            self.train_record_btn.configure(state="normal")
            self.train_stop_btn.configure(state="normal")
        else:
            self.train_record_btn.configure(state="disabled")
            self.train_stop_btn.configure(state="disabled")

    def start_recording_animation(self):
        """
        Show a progress bar animation while recording.
        """
        self.progress_bar.grid(row=3, column=0, padx=10, pady=(20, 0), columnspan=2, sticky="ew")
        self.progress_bar.start()

        while self.recording:
            time.sleep(0.02)

        self.progress_bar.stop()
        self.progress_bar.grid_forget()

    def show_widgets(self):
        """
        Show the widgets for the "Train Model" button.
        """
        self.train_input_label.grid(row=0, column=0, padx=10, pady=(20, 0), sticky="w")
        self.train_input_entry.grid(row=1, column=0, padx=10, pady=(20, 0), columnspan=2, sticky="ew")
        self.train_record_btn.grid(row=4, column=0, padx=10, pady=(20, 0), sticky="ew")
        self.train_stop_btn.grid(row=4, column=1, padx=10, pady=(20, 0), sticky="ew")

    def hide_widgets(self):
        """
        Hide the widgets for the "Train Model" button.
        """
        self.train_input_label.grid_forget()
        self.train_input_entry.grid_forget()
        self.train_record_btn.grid_forget()
        self.train_stop_btn.grid_forget()
        self.progress_bar.grid_forget()

    def init_audio_stream(self):
        """
        Initialize audio stream for recording.
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)
        return stream

    def record_audio(self):
        """
        Record audio and append frames.
        """
        while self.recording:
            data = self.audio_stream.read(1024)
            self.frames.append(data)

    def start_recording(self):
        """
        Start the recording process.
        """
        if not self.recording:
            self.recording = True
            threading.Thread(target=self.start_recording_animation).start()
            self.frames = []
            self.audio_stream = self.init_audio_stream()

            # Start a new thread for recording
            self.audio_thread = threading.Thread(target=self.record_audio)
            self.audio_thread.start()

    def stop_recording(self):
        """
        Stop the recording process.
        """
        if self.recording:
            self.recording = False
            self.audio_thread.join()  # Wait for the recording thread to finish
            self.save_audio()

    def setup_csv_file(self):
        """
        Initialize CSV metadata file for training dataset if it does not exist.
        """
        metadata = get_file_path(TRAIN_METADATA_DIR, TRAIN_METADATA_FILENAME)

        if not os.path.isfile(metadata):
            with open(metadata, 'w', newline='') as file:
                fields = ["file_name", "fold", "classID", "class"]
                writer = csv.writer(file)
                writer.writerow(fields)

    def save_audio(self):
        """
        Save the recorded audio.
        """
        self.audio_stream.stop_stream()
        self.audio_stream.close()

        p = pyaudio.PyAudio()

        self.setup_csv_file()

        user_name = self.train_input_entry_var.get().strip().lower()
        filename, user_id, audio_id = self.get_filename_info(user_name)

        # Construct the full file path
        file_path = get_file_path(TRAIN_DIR, filename)

        wf = wave.open(file_path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        log_data = {
            'file_name': filename,
            'fold': file_path,
            'classID': user_id,
            'class': user_name
        }

        log_into_csv(log_data)

        augment_audio(file_path, user_name, user_id, audio_id)

        self.train_model()

    def train_model(self):
        threading.Thread(target=start_training, args=(self.training_progress_bar, self)).start()

    def get_filename_info(self, user_name):
        """
        Generate unique filename for the recorded audio file.

        Parameters:
        - user_name (str): Name entered by the user.

        Audio files for training are located in dataset/train folder and have a naming format as:
        userid-username-audioid
        """
        # Get the metadata file
        metadata_path = get_file_path(TRAIN_METADATA_DIR, TRAIN_METADATA_FILENAME)
        metadata = pd.read_csv(metadata_path)

        # If len(metadata.index) is 0, then there are no audio recordings
        num_rows = len(metadata.index)

        if num_rows == 0:
            # No data in the CSV file
            user_id = 0
            audio_id = 0
            filename = f"{user_id}-{user_name}-{audio_id}.wav"
        elif (metadata['class'] == user_name).any():
            # User has some data in CSV
            row = metadata.loc[metadata['class'] == user_name].iloc[-1]
            audio_id = int(row["file_name"].replace(".wav", "").split("-")[2])
            user_id = row['classID']
            filename = f"{user_id}-{user_name}-{audio_id+1}.wav"
        else:
            # User does not have any data in CSV
            user_id = int(metadata['classID'].max())+1
            audio_id = 0
            filename = f"{user_id}-{user_name}-{audio_id}.wav"

        return filename, user_id, audio_id
