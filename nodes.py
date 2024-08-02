import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import folder_paths
import comfy.utils
import time
import copy
import math
import dill
import torch.nn.functional as torchfn
from torchvision import transforms
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(current_directory)
from .LivePortrait.src.live_portrait_wrapper import LivePortraitWrapper
from .LivePortrait.src.utils.rprint import rlog as log
from .LivePortrait.src.utils.camera import get_rotation_matrix
from .config.inference_config import InferenceConfig

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
def rgb_crop(rgb, region):
    return rgb[region[1]:region[3], region[0]:region[2]]

def rgb_crop_batch(rgbs, region):
    return rgbs[:, region[1]:region[3], region[0]:region[2]]
def get_rgb_size(rgb):
    return rgb.shape[1], rgb.shape[0]
def create_transform_matrix(x, y, scale=1):
    return np.float32([[scale, 0, x], [0, scale, y]])

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)

def calc_crop_limit(center, img_size, crop_size):
    pos = center - crop_size / 2
    if pos < 0:
        crop_size += pos * 2
        pos = 0

    pos2 = pos + crop_size

    if img_size < pos2:
        crop_size -= (pos2 - img_size) * 2
        pos2 = img_size
        pos = pos2 - crop_size

    return pos, pos2, crop_size

def retargeting(delta_out, driving_exp, factor, idxes):
    for idx in idxes:
        #delta_out[0, idx] -= src_exp[0, idx] * factor
        delta_out[0, idx] += driving_exp[0, idx] * factor

class PreparedSrcImg:
    def __init__(self, src_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori):
        self.src_rgb = src_rgb
        self.crop_trans_m = crop_trans_m
        self.x_s_info = x_s_info
        self.f_s_user = f_s_user
        self.x_s_user = x_s_user
        self.mask_ori = mask_ori

class LP_Engine:
    pipeline = None
    bbox_model = None
    mask_img = None

    def detect_face(self, image_rgb):

        crop_factor = 1.7
        bbox_drop_size = 10

        if self.bbox_model == None:
            bbox_model_path = os.path.join(get_model_dir("ultralytics"), "face_yolov8n.pt")
            self.bbox_model = YOLO(bbox_model_path)

        pred = self.bbox_model(image_rgb, conf=0.7, device="")
        bboxes = pred[0].boxes.xyxy.cpu().numpy()

        w, h = get_rgb_size(image_rgb)

        # for x, label in zip(segmasks, detected_results[0]):
        for x1, y1, x2, y2 in bboxes:
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            crop_w = bbox_w * crop_factor
            crop_h = bbox_h * crop_factor

            crop_w = max(crop_h, crop_w)
            crop_h = crop_w

            kernel_x = x1 + bbox_w / 2
            kernel_y = y1 + bbox_h / 2

            new_x1, new_x2, crop_w = calc_crop_limit(kernel_x, w, crop_w)

            if crop_w < crop_h:
                crop_h = crop_w

            new_y1, new_y2, crop_h = calc_crop_limit(kernel_y, h, crop_h)

            if crop_h < crop_w:
                crop_w = crop_h
                new_x1, new_x2, crop_w = calc_crop_limit(kernel_x, w, crop_w)

            return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

        print("Failed to detect face!!")
        return [0, 0, w, h]

    def crop_face(self, rgb_img):
        region = self.detect_face(rgb_img)
        face_image = rgb_crop(rgb_img, region)
        return face_image, region

    def get_pipeline(self):
        if self.pipeline == None:
            print("Load pipeline...")
            self.pipeline = LivePortraitWrapper(cfg=InferenceConfig())

        return self.pipeline

    def prepare_src_image(self, img):
        h, w = img.shape[:2]
        input_shape = [256,256]
        if h != input_shape[0] or w != input_shape[1]:
            x = cv2.resize(img, (input_shape[0], input_shape[1]), interpolation = cv2.INTER_LINEAR)
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.cuda()
        return x

    def GetMask(self):
        if self.mask_img is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./LivePortrait/src/utils/resources/mask_template.png")
            self.mask_img = cv2.imread(path, cv2.IMREAD_COLOR)
        return self.mask_img

    def prepare_source(self, source_image, is_video = False):
        print("Prepare source...")
        engine = self.get_pipeline()
        source_image_np = (source_image * 255).byte().numpy()
        img_rgb = source_image_np[0]
        face_img, crop_region = self.crop_face(img_rgb)

        scale = face_img.shape[0] / 512.
        crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], scale)
        mask_ori = cv2.warpAffine(self.GetMask(), crop_trans_m, get_rgb_size(img_rgb), cv2.INTER_LINEAR)
        mask_ori = mask_ori.astype(np.float32) / 255.

        psi_list = []
        for img_rgb in source_image_np:
            face_img = rgb_crop(img_rgb, crop_region)
            i_s = self.prepare_src_image(face_img)
            x_s_info = engine.get_kp_info(i_s)
            f_s_user = engine.extract_feature_3d(i_s)
            x_s_user = engine.transform_keypoint(x_s_info)
            psi = PreparedSrcImg(img_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori)
            if is_video == False:
                return psi
            psi_list.append(psi)

        return psi_list

    def prepare_driving_video(self, face_images):
        print("Prepare driving video...")
        pipeline = self.get_pipeline()
        f_img_np = (face_images * 255).byte().numpy()

        out_list = []
        for f_img in f_img_np:
            i_d = pipeline.prepare_source(f_img)
            d_info = pipeline.get_kp_info(i_d)
            #out_list.append((d_info, get_rotation_matrix(d_info['pitch'], d_info['yaw'], d_info['roll'])))
            out_list.append(d_info)

        return out_list

    def calc_fe(_, x_d_new, eyes, eyebrow, wink, pupil_x, pupil_y, mouth, eee, woo, smile,
                rotate_pitch, rotate_yaw, rotate_roll):

        x_d_new[0, 20, 1] += smile * -0.01
        x_d_new[0, 14, 1] += smile * -0.02
        x_d_new[0, 17, 1] += smile * 0.0065
        x_d_new[0, 17, 2] += smile * 0.003
        x_d_new[0, 13, 1] += smile * -0.00275
        x_d_new[0, 16, 1] += smile * -0.00275
        x_d_new[0, 3, 1] += smile * -0.0035
        x_d_new[0, 7, 1] += smile * -0.0035

        x_d_new[0, 19, 1] += mouth * 0.001
        x_d_new[0, 19, 2] += mouth * 0.0001
        x_d_new[0, 17, 1] += mouth * -0.0001
        rotate_pitch -= mouth * 0.05

        x_d_new[0, 20, 2] += eee * -0.001
        x_d_new[0, 20, 1] += eee * -0.001
        #x_d_new[0, 19, 1] += eee * 0.0006
        x_d_new[0, 14, 1] += eee * -0.001

        x_d_new[0, 14, 1] += woo * 0.001
        x_d_new[0, 3, 1] += woo * -0.0005
        x_d_new[0, 7, 1] += woo * -0.0005
        x_d_new[0, 17, 2] += woo * -0.0005

        x_d_new[0, 11, 1] += wink * 0.001
        x_d_new[0, 13, 1] += wink * -0.0003
        x_d_new[0, 17, 0] += wink * 0.0003
        x_d_new[0, 17, 1] += wink * 0.0003
        x_d_new[0, 3, 1] += wink * -0.0003
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            x_d_new[0, 11, 0] += pupil_x * 0.0007
            x_d_new[0, 15, 0] += pupil_x * 0.001
        else:
            x_d_new[0, 11, 0] += pupil_x * 0.001
            x_d_new[0, 15, 0] += pupil_x * 0.0007

        x_d_new[0, 11, 1] += pupil_y * -0.001
        x_d_new[0, 15, 1] += pupil_y * -0.001
        eyes -= pupil_y / 2.

        x_d_new[0, 11, 1] += eyes * -0.001
        x_d_new[0, 13, 1] += eyes * 0.0003
        x_d_new[0, 15, 1] += eyes * -0.001
        x_d_new[0, 16, 1] += eyes * 0.0003


        if 0 < eyebrow:
            x_d_new[0, 1, 1] += eyebrow * 0.001
            x_d_new[0, 2, 1] += eyebrow * -0.001
        else:
            x_d_new[0, 1, 0] += eyebrow * -0.001
            x_d_new[0, 2, 0] += eyebrow * 0.001
            x_d_new[0, 1, 1] += eyebrow * 0.0003
            x_d_new[0, 2, 1] += eyebrow * -0.0003


        return torch.Tensor([rotate_pitch, rotate_yaw, rotate_roll])
g_engine = LP_Engine()

class ExpressionSet:
    def __init__(self, erst = None, es = None):
        if es != None:
            self.e = copy.deepcopy(es.e)  # [:, :, :]
            self.r = copy.deepcopy(es.r)  # [:]
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst != None:
            self.e = erst[0]
            self.r = erst[1]
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.from_numpy(np.zeros((1, 21, 3))).float().to(device='cuda')
            self.r = torch.Tensor([0, 0, 0])
            self.s = 0
            self.t = 0
    def div(self, value):
        self.e /= value
        self.r /= value
        self.s /= value
        self.t /= value
    def add(self, other):
        self.e += other.e
        self.r += other.r
        self.s += other.s
        self.t += other.t
    def sub(self, other):
        self.e -= other.e
        self.r -= other.r
        self.s -= other.s
        self.t -= other.t
    def mul(self, value):
        self.e *= value
        self.r *= value
        self.s *= value
        self.t *= value

    #def apply_ratio(self, ratio):        self.exp *= ratio

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time - start_time))
        return result

    return wrapper_fn


exp_data_dir = os.path.join(current_directory, "exp_data")
class SaveExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "file_name": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {"save_exp": ("EXP_DATA",), }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_name",)
    FUNCTION = "run"
    CATEGORY = "파워집돌이"
    OUTPUT_NODE = True

    def run(self, file_name, save_exp:ExpressionSet=None):
        if save_exp == None or file_name == "":
            return file_name

        with open(os.path.join(exp_data_dir, file_name + ".exp"), "wb") as f:
            dill.dump(save_exp, f)

        return file_name

class LoadExpData:
    @classmethod
    def INPUT_TYPES(s):
        file_list = [os.path.splitext(file)[0] for file in os.listdir(exp_data_dir) if file.endswith('.exp')]
        return {"required": {
            "file_name": (sorted(file_list, key=str.lower),),
            "ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "파워집돌이"

    def run(self, file_name, ratio):
        # es = ExpressionSet()
        with open(os.path.join(exp_data_dir, file_name + ".exp"), 'rb') as f:
            es = dill.load(f)
        es.mul(ratio)
        return (es,)

class ExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                #"code": ("STRING", {"multiline": False, "default": ""}),
                "code1": ("INT", {"default": 0}),
                "value1": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "code2": ("INT", {"default": 0}),
                "value2": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "code3": ("INT", {"default": 0}),
                "value3": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "code4": ("INT", {"default": 0}),
                "value4": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "code5": ("INT", {"default": 0}),
                "value5": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
            },
            "optional":{"add_exp": ("EXP_DATA",),}
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "파워집돌이"

    def run(self, code1, value1, code2, value2, code3, value3, code4, value4, code5, value5, add_exp=None):
        #print(f"type(None):{type(None)}")
        #if type(add_exp) == type(None):
        if add_exp == None:
            es = ExpressionSet()
            log(f"exp11:{es.exp[0,1,1]}")
        else:
            es = ExpressionSet(es = add_exp)
            if id(es.exp) == id(add_exp.exp):
                log("id(es.exp) == id(add_exp.exp) is True")

        codes = [code1, code2, code3, code4, code5]
        values = [value1, value2, value3, value4, value5]
        for i in range(5):
            idx = int(codes[i] / 10)
            r = codes[i] % 10
            es.exp[0, idx, r] += values[i] * 0.001

        return (es,)

class PrintExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "cut_noise": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 0.1}),
        },
            "optional": {"exp": ("EXP_DATA",), }
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "파워집돌이"
    OUTPUT_NODE = True

    def run(self, cut_noise, exp = None):
        if exp == None: return (exp,)

        cuted_list = []
        e = exp.exp * 1000
        for idx in range(21):
            for r in range(3):
                a = abs(e[0, idx, r])
                if(cut_noise < a): cuted_list.append((a, e[0, idx, r], idx*10+r))

        sorted_list = sorted(cuted_list, reverse=True, key=lambda item: item[0])
        print(f"sorted_list: {[[item[2], round(float(item[1]),1)] for item in sorted_list]}")
        return (exp,)

class Command:
    def __init__(self, es, change, keep):
        self.es:ExpressionSet = es
        self.change = change
        self.keep = keep
class AdvancedLivePortrait:
    def __init__(s):
        s.src_images = None
        s.driving_images = None
        s.pbar = comfy.utils.ProgressBar(1)

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "retargeting_eyes": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "retargeting_mouth": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "turn_on": ("BOOLEAN", {"default": True}),
                "command": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "src_images": ("IMAGE",),
                "motion_link": ("EDITOR_LINK",),
                "driving_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "파워집돌이"

    # INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (False,)

    def parsing_command(self, command, motoin_link):
        command.replace(' ', '')
        # if command == '': return
        lines = command.split('\n')

        cmd_list = []

        total_length = 0

        i = 0
        #old_es = None
        for line in lines:
            i += 1
            if line == '': continue
            try:
                cmds = line.split('=')
                #assert len(cmds) == 2, f"(파워집돌이) 명령어오류 {i}번줄: {line}: '=' 기호가 1개 들어가야 합니다"
                idx = int(cmds[0])
                if idx == 0: es = ExpressionSet()
                else: es = ExpressionSet(es = motoin_link[idx])
                cmds = cmds[1].split(':')
                change = int(cmds[0])
                keep = int(cmds[1])
            except:
                #log(f"(파워집돌이) 명령어오류 {i}번줄: {line}")
                assert False, f"(파워집돌이) 명령어오류 {i}번줄: {line}"


                return None, None

            total_length += change + keep
            es.div(change)
            cmd_list.append(Command(es, change, keep))

        return cmd_list, total_length


    def run(self, retargeting_eyes, retargeting_mouth, turn_on, command,
            src_images=None, driving_images=None, motion_link=None):
        if turn_on == False: return (None,None)
        src_length = 1

        if src_images == None:
            if motion_link != None:
                self.psi_list = [motion_link[0]]
            else: return (None,None)

        if src_images != None:
            src_length = len(src_images)
            if id(src_images) != id(self.src_images):
                self.src_images = src_images
                if 1 < src_length:
                    self.psi_list = g_engine.prepare_source(src_images, True)
                else:
                    self.psi_list = [g_engine.prepare_source(src_images)]


        cmd_list, cmd_length = self.parsing_command(command, motion_link)
        if cmd_list == None: return (None,None)
        cmd_idx = 0

        driving_length = 0
        if driving_images is not None:
            if id(driving_images) != id(self.driving_images):
                self.driving_images = driving_images
                self.driving_values = g_engine.prepare_driving_video(driving_images)
            driving_length = len(self.driving_values)

        #total_length = max(driving_length, cmd_length, src_length)
        total_length = max(driving_length, src_length)

        c_i_es = ExpressionSet()
        c_o_es = ExpressionSet()
        d_0_es = None
        out_list = []

        psi = None
        pipeline = g_engine.get_pipeline()
        for i in range(total_length):

            if i < src_length:
                psi = self.psi_list[i]
                s_info = psi.x_s_info
                s_es = ExpressionSet(erst=(s_info['kp'] + s_info['exp'], torch.Tensor([0, 0, 0]), s_info['scale'], s_info['t']))

            new_es = ExpressionSet(es = s_es)

            if i < cmd_length:
                cmd = cmd_list[cmd_idx]
                #cmd = Command()#지울거
                if 0 < cmd.change:
                    cmd.change -= 1
                    c_i_es.add(cmd.es)
                    c_i_es.sub(c_o_es)
                elif 0 < cmd.keep:
                    cmd.keep -= 1

                new_es.add(c_i_es)

                if cmd.change == 0 and cmd.keep == 0:
                    cmd_idx += 1
                    if cmd_idx < len(cmd_list):
                        c_o_es = ExpressionSet(es = c_i_es)
                        cmd = cmd_list[cmd_idx]
                        c_o_es.div(cmd.change)
            elif 0 < cmd_length:
                new_es.add(c_i_es)

            if i < driving_length:
                d_i_info = self.driving_values[i]
                d_i_r = torch.Tensor([d_i_info['pitch'], d_i_info['yaw'], d_i_info['roll']])#.float().to(device="cuda:0")

                if d_0_es is None:
                    d_0_es = ExpressionSet(erst = (d_i_info['exp'], d_i_r, d_i_info['scale'], d_i_info['t']))
                    #d_i_es = d_0_es
                    retargeting(s_es.e, d_0_es.e, retargeting_eyes, (11, 13, 15, 16))
                    retargeting(s_es.e, d_0_es.e, retargeting_mouth, (14, 17, 19, 20))

                #r_new = (r_d_i @ d_0_es.r.permute(0, 2, 1)) @ r_new
                new_es.e += d_i_info['exp'] - d_0_es.e
                new_es.r += d_i_r - d_0_es.r
                new_es.t += d_i_info['t'] - d_0_es.t

            r_new = get_rotation_matrix(
                s_info['pitch'] + new_es.r[0], s_info['yaw'] + new_es.r[1], s_info['roll'] + new_es.r[2])
            d_new = new_es.s * (new_es.e @ r_new) + new_es.t
            d_new = pipeline.stitching(psi.x_s_user, d_new)
            crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, d_new)
            crop_out = pipeline.parse_output(crop_out['out'])[0]

            crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb),
                                                cv2.INTER_LINEAR)
            out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(
                np.uint8)
            out_list.append(out)

            self.pbar.update_absolute(i+1, total_length, ("PNG", Image.fromarray(crop_out), None))

        if len(out_list) == 0: return (None,)

        out_imgs = torch.cat([pil2tensor(img_rgb) for img_rgb in out_list])
        return (out_imgs,)

class ExpressionEditor:
    def __init__(self):
        self.sample_image = None
        self.src_image = None

    @classmethod
    def INPUT_TYPES(s):
        display = "number"
        #display = "slider"
        return {
            "required": {

                "rotate_pitch": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),
                "rotate_yaw": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),
                "rotate_roll": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),

                "blink": ("FLOAT", {"default": 0, "min": -20, "max": 5, "step": 0.5, "display": display}),
                "eyebrow": ("FLOAT", {"default": 0, "min": -10, "max": 15, "step": 0.5, "display": display}),
                "wink": ("FLOAT", {"default": 0, "min": 0, "max": 25, "step": 0.5, "display": display}),
                "pupil_x": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.5, "display": display}),
                "pupil_y": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.5, "display": display}),
                "aaa": ("FLOAT", {"default": 0, "min": -30, "max": 120, "step": 1, "display": display}),
                "eee": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
                "woo": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
                "smile": ("FLOAT", {"default": 0, "min": -0.3, "max": 1.3, "step": 0.01, "display": display}),

                "src_ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "display": display}),
                "sample_ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "display": display}),
            },

            "optional": {"src_image": ("IMAGE",), "motion_link": ("EDITOR_LINK",),
                         "sample_image": ("IMAGE",), "add_exp": ("EXP_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE", "EDITOR_LINK", "EXP_DATA")
    RETURN_NAMES = ("image", "motion_link", "save_exp")

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "파워집돌이"

    # INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (False,)

    def run(self, rotate_pitch, rotate_yaw, rotate_roll, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
            src_ratio, sample_ratio, src_image=None, sample_image=None, motion_link=None, add_exp=None):
        rotate_yaw = -rotate_yaw

        new_editor_link = None
        if motion_link != None:
            self.psi = motion_link[0]
            new_editor_link = motion_link.copy()
        elif src_image != None:
            if id(src_image) != id(self.src_image):
                self.psi = g_engine.prepare_source(src_image)
                self.src_image = src_image
            new_editor_link = []
            new_editor_link.append(self.psi)
        else:
            return (None,None)

        pipeline = g_engine.get_pipeline()

        psi = self.psi
        s_info = psi.x_s_info
        #delta_new = copy.deepcopy()
        s_exp = s_info['exp'] * src_ratio
        s_exp[0, 5] = s_info['exp'][0, 5]
        s_exp += s_info['kp']

        es = ExpressionSet()

        if sample_image != None:
            if id(self.sample_image) != id(sample_image):
                self.sample_image = sample_image
                d_image_np = (sample_image * 255).byte().numpy()
                d_face, _ = g_engine.crop_face(d_image_np[0])
                i_d = pipeline.prepare_source(d_face)
                self.d_info = pipeline.get_kp_info(i_d)
                self.d_info['exp'][0, 5, 0] = 0
                self.d_info['exp'][0, 5, 1] = 0

            # delta_new += s_exp * (1 - sample_ratio) + self.d_info['exp'] * sample_ratio
            es.e += self.d_info['exp'] * sample_ratio

        es.r = g_engine.calc_fe(es.e, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
                                  rotate_pitch, rotate_yaw, rotate_roll)

        if add_exp != None:
            es.add(add_exp)

        new_rotate = get_rotation_matrix(s_info['pitch'] + es.r[0], s_info['yaw'] + es.r[1],
                                         s_info['roll'] + es.r[2])
        x_d_new = (s_info['scale'] * (1 + es.s)) * ((s_exp + es.e) @ new_rotate) + s_info['t']

        x_d_new = pipeline.stitching(psi.x_s_user, x_d_new)

        crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
        crop_out = pipeline.parse_output(crop_out['out'])[0]

        crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
        out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)

        out_img = pil2tensor(out)

        filename = "fe_edit_preview.png"
        folder_paths.get_save_image_path(filename, folder_paths.get_temp_directory())
        img = Image.fromarray(crop_out)
        img.save(os.path.join(folder_paths.get_temp_directory(), filename), compress_level=1)
        results = list()
        results.append({"filename": filename, "type": "temp"})

        new_editor_link.append(es)

        return {"ui": {"images": results}, "result": (out_img, new_editor_link, es)}

class TestNode:

    def __init__(s):
        s.pbar = comfy.utils.ProgressBar(1)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            #"images": ("IMAGE",),
            #"test_value": ("INT", {"default": 1, "min": 1}),
            "command": ("STRING", {"multiline": True, "default": ""}),
        },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    def run(self, command):
        self.parsing_command(command)

        return (None,)

NODE_CLASS_MAPPINGS = {
    "AdvancedLivePortrait": AdvancedLivePortrait,
    "ExpressionEditor": ExpressionEditor,
    "ExpData": ExpData,
    "LoadExpData": LoadExpData,
    "SaveExpData": SaveExpData,
    "PrintExpData:": PrintExpData,
    #"TestNode": TestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLivePortrait": "Advanced Live Portrait (파워집돌이)",
    "ExpressionEditor": "Expression Editor (파워집돌이)",
}