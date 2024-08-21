# ComfyUI-AdvancedLivePortrait

## Update

8/21/2024

You can create a video without a video.

Track the face of the source video.

The workflow has been updated.

## Introduction

AdvancedLivePortrait is faster and has real-time preview

https://github.com/user-attachments/assets/90b78639-6477-48af-ba49-7945488df581

Edit facial expressions in photos.

Insert facial expressions into videos.

Create animations using multiple facial expressions.

Extract facial expressions from sample photos.

## Installation

This project has been registered with ComfyUI-Manager. Now you can install it automatically using the manager.

## Usage

The workflows and sample datas placed in '\custom_nodes\ComfyUI-AdvancedLivePortrait\sample\'

-----

You can add expressions to the video. See 'workflow2_advanced.json'.

Describes the 'command' in 'workflow2_advanced.json'

![readme](https://github.com/user-attachments/assets/339568b2-ad52-4aaf-a6ab-fcd877449c56)


[Motion index] = [Changing frame length] : [Length of frames waiting for next motion]

Motion index 0 is the original source image.

They are numbered in the order they lead to the motion_link.

Linking the driving video to 'src_images' will add facial expressions to the driving video.

-----

You can save and load expressions with the 'Load Exp Data' 'Save Exp Data' nodes.

\ComfyUI\output\exp_data\  Path to the folder being saved

-----

## Thanks

Original author's link : https://liveportrait.github.io/

This project uses a model converted by kijai. link : https://github.com/kijai/ComfyUI-LivePortraitKJ
