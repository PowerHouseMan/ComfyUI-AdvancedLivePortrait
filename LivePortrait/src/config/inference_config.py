
import os
from typing import Literal, Tuple
import folder_paths

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
class InferenceConfig:
    def __init__(self):
        self.models_config: str = os.path.join(current_directory, "models.yaml")
        self.checkpoint_F: str = os.path.join(folder_paths.models_dir, "liveportrait", "base_models", "appearance_feature_extractor.pth")
        self.checkpoint_M: str = os.path.join(folder_paths.models_dir, "liveportrait", "base_models", "motion_extractor.pth")
        self.checkpoint_G: str = os.path.join(folder_paths.models_dir, "liveportrait", "base_models", "spade_generator.pth")
        self.checkpoint_W: str = os.path.join(folder_paths.models_dir, "liveportrait", "base_models", "warping_module.pth")
        self.checkpoint_S: str = os.path.join(folder_paths.models_dir, "liveportrait", "retargeting_models", "stitching_retargeting_module.pth")
        self.flag_use_half_precision: bool = True  # whether to use half precision

        self.input_shape: Tuple[int, int] = (256, 256)  # input shape

        self.mask_crop = None
        self.device_id: int = 0