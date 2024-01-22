import os
import customtkinter
from ml_scripts.utils import get_file_path, MODEL_SAVE_PATH, MODEL_SAVE_NAME, truncate_metadata, \
    delete_train_dataset


class ResetWidgets:
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

    def reset_training(self):
        """
        Clear training data by:
        - truncating train_metadata.csv
        - deleting audio files in the dataset/train folder
        - deleting saved model
        """
        truncate_metadata()
        delete_train_dataset()

        saved_model = get_file_path(MODEL_SAVE_PATH, MODEL_SAVE_NAME)
        if os.path.isfile(saved_model):
            os.remove(saved_model)

        self.status_label.grid(row=0, column=0, padx=10, pady=(20, 0), sticky="w")

