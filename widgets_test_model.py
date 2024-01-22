import random
import time
import wave
from ml_scripts.inference import predict
import customtkinter
import threading
from ml_scripts.utils import *
import pyaudio


class TestModelWidgets:
    def __init__(self, parent_frame):
        """
        Initialize the TestModelWidgets class.

        Parameters:
        - parent_frame (tk.Frame): Parent frame for the widgets.
        """
        self.parent_frame = parent_frame
        self.recording = False

        self.create_widgets()

        # Audio recording variables
        self.frames = []
        self.audio_stream = None
        self.audio_thread = None

    def create_widgets(self):
        """
        Create widgets for the "Test Model" button.
        Initially, they are hidden.
        """
        self.test_input_label = customtkinter.CTkLabel(self.parent_frame,
                                                        text="Test the model",
                                                        font=("Times", 18, "bold"))

        self.test_record_btn = customtkinter.CTkButton(self.parent_frame,
                                                        text="Start recording",
                                                        fg_color="#006400",
                                                        hover_color="#008000")

        self.test_stop_btn = customtkinter.CTkButton(self.parent_frame,
                                                      text="Stop recording",
                                                      fg_color="#343a40",
                                                      hover_color="#495057")

        self.progress_bar = customtkinter.CTkProgressBar(self.parent_frame,
                                                         width=200,
                                                         mode="indeterminate")

        self.testing_progress_bar = customtkinter.CTkProgressBar(self.parent_frame,
                                                                  width=200,
                                                                  mode="determinate")

        self.prediction_label = customtkinter.CTkLabel(self.parent_frame,
                                                       text="",
                                                       font=("Times", 18, "bold"))

        self.hide_widgets()

        self.test_record_btn.configure(command=self.start_recording)
        self.test_stop_btn.configure(command=self.stop_recording)

    def show_widgets(self):
        """
        Show the widgets for the "Test Model" button.
        """
        self.test_input_label.grid(row=0, column=0, padx=10, pady=(20, 0), sticky="w")
        self.test_record_btn.grid(row=5, column=0, padx=10, pady=(200, 0), sticky="ew")
        self.test_stop_btn.grid(row=5, column=1, padx=10, pady=(200, 0), sticky="ew")

    def hide_widgets(self):
        """
        Hide the widgets for the "Test Model" button.
        """
        self.test_record_btn.grid_forget()
        self.test_stop_btn.grid_forget()
        self.progress_bar.grid_forget()
        self.testing_progress_bar.place_forget()
        self.prediction_label.place_forget()
        self.test_input_label.grid_forget()

    def start_recording(self):
        """
        Start the recording process.
        """
        if not self.recording:
            self.prediction_label.place_forget()
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

    def start_recording_animation(self):
        """
        Show a progress bar animation while recording.
        """
        self.progress_bar.place(x=60, y=180)
        self.progress_bar.start()

        while self.recording:
            time.sleep(0.02)

        self.progress_bar.stop()
        self.progress_bar.place_forget()

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

    def save_audio(self):
        """
        Save the recorded audio.
        """
        self.audio_stream.stop_stream()
        self.audio_stream.close()

        p = pyaudio.PyAudio()

        # Generate name for the test audio file

        file_name = "test.wav"
        file_path = get_file_path(TEST_DIR, file_name)

        wf = wave.open(file_path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        self.test_model(file_path)

    def test_model(self, file_path):
        threading.Thread(target=predict, args=(self.testing_progress_bar, self, file_path)).start()
