import requests
import os, sys
import subprocess
from tqdm import tqdm
from pip._internal import main as pip_main
from pathlib import Path
from folder_paths import models_dir

def download_model(file_path, model_url):
    print('AdvancedLivePortrait: Downloading model...')
    response = requests.get(model_url, stream=True)
    try:
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            # tqdm will display a progress bar
            with open(file_path, 'wb') as file, tqdm(
                desc='Downloading',
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)

    except requests.exceptions.RequestException as err:
        print('AdvancedLivePortrait: Model download failed: {err}')
        print(f'AdvancedLivePortrait: Download it manually from: {model_url}')
        print(f'AdvancedLivePortrait: And put it in {file_path}')
    except Exception as e:
        print(f'AdvancedLivePortrait: An unexpected error occurred: {e}')

save_path = os.path.join(models_dir, "ultralytics")
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "face_yolov8n.pt")
    if not Path().is_file():
        download_model(file_path, "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt")
    

    

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


