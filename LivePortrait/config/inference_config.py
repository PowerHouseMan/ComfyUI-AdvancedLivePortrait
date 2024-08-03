
import os
from typing import Literal, Tuple
import folder_paths

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
class InferenceConfig:
    def __init__(self):
        self.flag_use_half_precision: bool = True  # whether to use half precision
        self.input_shape: Tuple[int, int] = (256, 256)  # input shape

        self.mask_crop = None
        self.device_id: int = 0