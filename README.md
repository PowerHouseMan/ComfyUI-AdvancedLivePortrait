https://github.com/user-attachments/assets/90b78639-6477-48af-ba49-7945488df581

\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-AdvancedLivePortrait\install.bat
Run it.


https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib
Do not use insightface.
You only need 'liveportait' folder.
\ComfyUI_windows_portable\ComfyUI\models\
Put it here.


\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-AdvancedLivePortrait\sample\
There are workflows and sample data.


You can add expressions to the video. See sample2.json workflow.

Command description in sample2.json.
1 = 1:10
2 = 5:10
0 = 2:50
1 = 2:0
[Motion index] = [Changing frame length] : [Length of frames waiting for next motion]
Motion index 0 is the original source image.
They are numbered in the order they lead to the motion_link.
Linking a driving video to src_images will add facial expressions to the driving video.

Thanks

Original author's link https://liveportrait.github.io/
