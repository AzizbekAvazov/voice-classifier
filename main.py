import csv
import os
import customtkinter

from ml_scripts.utils import APP_NAME, MAIN_WINDOW_HEIGHT_PERCENT, MAIN_WINDOW_WIDTH_PERCENT
from widgets_train_model import TrainModelWidgets
from widgets_test_model import TestModelWidgets
from widgets_reset import ResetWidgets


class VoiceRecorderApp:
    def __init__(self, master):
        self.master = master
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()

        self.setup_main_window()
        self.center_window()

        # Create Left frame
        self.create_left_frame()

        # Create Right frame
        self.create_right_frame()

        self.create_train_model_widgets()
        self.create_test_model_widgets()
        self.create_reset_widgets()

    def setup_main_window(self):
        """
        Configure main window properties.
        """
        self.master.title(APP_NAME)
        window_width = int(self.screen_width * (MAIN_WINDOW_WIDTH_PERCENT / 100))
        window_height = int(self.screen_height * (MAIN_WINDOW_HEIGHT_PERCENT / 100))
        self.master.geometry(f"{window_width}x{window_height}")
        self.master.resizable(False, False)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

    def center_window(self):
        """
        Center the main window on the screen.
        """
        self.master.update_idletasks()
        width = self.master.winfo_width()
        frm_width = self.master.winfo_rootx() - self.master.winfo_x()
        win_width = width + 2 * frm_width
        height = self.master.winfo_height()
        titlebar_height = self.master.winfo_rooty() - self.master.winfo_y()
        win_height = height + titlebar_height + frm_width
        x = self.master.winfo_screenwidth() // 2 - win_width // 2
        y = self.master.winfo_screenheight() // 2 - win_height // 2
        self.master.geometry(f"{width}x{height}+{x}+{y}")
        self.master.deiconify()

    def create_left_frame(self):
        """
        Configure the left frame:
        - its location
        - its content
        """
        self.left_frame = customtkinter.CTkFrame(self.master, corner_radius=0)
        self.left_frame.grid(row=0, column=0, rowspan=7, sticky="nsew")
        self.left_frame.grid_rowconfigure(7, weight=1)

        buttons = [
            ("Train Model", "train_model"),
            ("Test Model", "test_model"),
            ("Reset", "reset")
        ]

        for row, (button_text, command) in enumerate(buttons):
            btn = customtkinter.CTkButton(self.left_frame, text=button_text, command=lambda cmd=command: self.handle_button_click(cmd))
            btn.grid(row=row, column=0, padx=20, pady=(20, 10))

    def create_right_frame(self):
        """
        Configure the right frame.
        """
        self.right_frame = customtkinter.CTkFrame(self.master, corner_radius=0)
        self.right_frame.grid(row=0, column=1, rowspan=1, sticky="nsew")

    def create_train_model_widgets(self):
        """
        Create widgets for the "Train Model" button.
        Initially, they are hidden.
        """
        self.train_model_widgets = TrainModelWidgets(self.right_frame)

    def create_test_model_widgets(self):
        """
        Create widgets for the "Test Model" button.
        Initially, they are hidden.
        """
        self.test_model_widgets = TestModelWidgets(self.right_frame)

    def create_reset_widgets(self):
        """
        Create widgets for the "Reset" button.
        Initially, they are hidden.
        """
        self.reset_widgets = ResetWidgets(self.right_frame)

    def handle_button_click(self, command):
        """
        Handle button clicks

        Parameters:
        - command: The command associated with the clicked button
        """
        if command == "train_model":
            self.train_model_widgets.show_widgets()
            self.test_model_widgets.hide_widgets()
            self.reset_widgets.hide_widgets()
        elif command == "test_model":
            self.test_model_widgets.show_widgets()
            self.train_model_widgets.hide_widgets()
            self.reset_widgets.hide_widgets()
        elif command == "reset":
            self.reset_widgets.reset_training()
            self.test_model_widgets.hide_widgets()
            self.train_model_widgets.hide_widgets()
        else:
            self.test_model_widgets.hide_widgets()
            self.train_model_widgets.hide_widgets()
            self.reset_widgets.hide_widgets()


def main():
    # Create main window 'master'
    master = customtkinter.CTk()

    # Initialize VoiceRecorderApp
    VoiceRecorderApp(master)

    # Start Tkinter Event loop
    master.mainloop()


if __name__ == '__main__':
    main()
