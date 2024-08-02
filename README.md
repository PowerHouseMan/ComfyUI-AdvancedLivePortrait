https://github.com/user-attachments/assets/90b78639-6477-48af-ba49-7945488df581


--- install ---

\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-AdvancedLivePortrait\install.bat  Run it.

https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib  download

Do not use insightface.

You only need 'liveportait' folder.

\ComfyUI_windows_portable\ComfyUI\models\  Put it here.

-----

\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-AdvancedLivePortrait\sample\  There are workflows and sample data.


You can add expressions to the video. See sample2.json workflow.

Describes the 'command' in sample2.json.

![readme](https://github.com/user-attachments/assets/339568b2-ad52-4aaf-a6ab-fcd877449c56)


[Motion index] = [Changing frame length] : [Length of frames waiting for next motion]

Motion index 0 is the original source image.

They are numbered in the order they lead to the motion_link.

Linking the driving video to 'src_images' will add facial expressions to the driving video.

-----

You can save and load expressions with the 'LoadExpData' 'SaveExpData' nodes.

-----

Thanks

Original author's link https://liveportrait.github.io/
