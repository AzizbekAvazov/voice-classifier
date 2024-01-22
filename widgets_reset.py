import os
import customtkinter
from ml_scripts.utils import get_file_path, TRAIN_METADATA_DIR, TRAIN_METADATA_FILENAME, resource_path, TRAIN_DIR


class ClearTrainDataWidgets:
    def __init__(self, parent_frame):
        """
        Initialize the ClearTrainDataWidgets class.

        Parameters:
        - parent_frame (tk.Frame): Parent frame for the widgets.
        """
        self.parent_frame = parent_frame
        self.create_widgets()

    def create_widgets(self):
        """
        Create widgets for the "Clear Training Data" button.
        """
        # label to show status after training data was cleared
        self.status_label = customtkinter.CTkLabel(self.parent_frame,
                                                   text="All training data has been cleared",
                                                   font=("Times", 18, "bold"))

        # widgets are hidden initially
        self.hide_widgets()

    def show_widgets(self):
        """
        Show the widgets for the "Clear Training Data" button.
        """
        self.status_label.grid(row=0, column=0, padx=10, pady=(20, 0), sticky="w")

    def hide_widgets(self):
        """
        Hide the widgets for the "Clear Training Data" button.
        """
        self.status_label.grid_forget()

    def clear_training_data(self):
        """
        Clear training data by truncating train_metadata.csv and deleting audio files in the dataset/train folder.
        """
        # clear the train_metadata.csv
        metadata_file = get_file_path(TRAIN_METADATA_DIR, TRAIN_METADATA_FILENAME)

        if os.path.isfile(metadata_file):
            with open(metadata_file, "r+") as file:
                file.readline()
                file.truncate(file.tell())

        # delete audio files located in dataset/train
        train_folder = resource_path(TRAIN_DIR)
        try:
            for filename in os.listdir(train_folder):
                os.remove(f"{train_folder}/{filename}")
        except FileNotFoundError:
            pass

        self.status_label.grid(row=0, column=0, padx=10, pady=(20, 0), sticky="w")
