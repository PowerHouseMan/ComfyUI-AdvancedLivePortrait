
import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
class InferenceConfig:
    def __init__(self):
        self.flag_use_half_precision: bool = False  # whether to use half precision