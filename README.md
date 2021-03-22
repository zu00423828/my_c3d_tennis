# my_c3d_tennis

C3D 網球分類 使用C3D模型進行軌跡分類
Tensorflow(>=1.4)
Python(>=2.7)
## Dataset
UCF101. You need to place it in the root directory of the workspace.
## Usage
```Bash
sudo ./convert_video_to_images.sh UCF101/ 5
```
to convert videos into images(5FPS per-second).
```Bash
./convert_images_to_list.sh UCF101/ 4
```
to obtain train and test sets.(3/4 train; 1/4 test)
```Bash
python train.py
```
for training.
```Bash
python test.py
```
