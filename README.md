# My-Diff
 It is based on [mmdetection](https://github.com/open-mmlab/mmdetection/tree/2.x) (v2.28.2).

## Models and Results
|Method|Backbone|Pretrain|$AP$|$AP_{50}$|$AP_{75}$|Model|
|:-|:-|:-|:-|:-|:-|:-|
|EnDiff-r50|ResNet50|[cascade_rcnn_r50_coco2017](https://github.com/dingdongtu521/en-diff/releases/download/Models/cascade_rcnn_r50_coco2017.pth)|49.9|82.8|52.6|[endiff_r50_urpc](https://github.com/dingdongtu521/en-diff/releases/download/Models/endiff_r50_urpc.pth)|
|EnDiff-xt101|ResNetXT101|[cascade_rcnn_xt101_coco2017](https://github.com/dingdongtu521/en-diff/releases/download/Models/cascade_rcnn_xt101_coco2017.pth)|50.5|84.1|54.4|[endiff_xt101_urpc](https://github.com/dingdongtu521/en-diff/releases/download/Models/endiff_xt101_urpc.pth)|


## Usage
### Installing
To create a new environment, run:
```shell
conda create -n endiff python=3.10 -y
conda activate endiff
```
To install pytorch run:
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
```
To install mmdetection, run:
```shell
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html 

pip install yapf==0.40.1 numpy==1.26.4 mmdet==2.28.2
```
To clone EnDiff, run:
```shell
git clone https://github.com/BitStrawber/my_diff.git
cd en-diff
```
### Data Preperation
The data should be orginized as follow:
```
en-diff/
    data/
        mydata/
            annotations/
            images/
        enhance/
```
- URPC2020 can be downloaded from [here](https://github.com/xiaoDetection/Learning-Heavily-Degraded-Prior/releases/download/datasets/urpc2020.zip).
- COCO2017 can be downloaded from [here](https://cocodataset.org/#download)

### Testing
Here we take testing `EnDiff-r50` as an example.

First download our checkpoint file to `checkpoints/endiff_r50_urpc.pth`:
```shell
mkdir checkpoints
wget -P ./checkpoints/ https://github.com/https://github.com/dingdongtu521/en-diff/releases/download/Models/endiff_r50_urpc.pth
```
Then test our model (set '--cfg-options' to avoid loading pre-trained weights):
```shell
python tools/test.py \
    configs/EnDiff_r50_diff.py \
    ./checkpoints/endiff_r50_urpc.pth \
    --eval bbox \
    --cfg-options model.init_cfg=None
```

### Fusion

We make use of groundedsam to generate your fusion_iamges.


```shell
#change the input_root and output_root as yours
cd path/to/your/Grounded-Segment-Anything
python TEST.py
```
you can get masks orginized as :
```
your_dataset/
    class1/
        images/
        masks/
    class2/
        images/
        masks/
    ...
```
then run:
```shell
python mulit_fusion_new.py
```
to generate dataset organized as:
```
output_root/
    blended_images/
    annotations/
    visualization/
```
### Training

Fist download our pre-trained model:
```shell
wget -P ./checkpoints/ https://github.com/dingdongtu521/en-diff/releases/download/Models/cascade_rcnn_r50_coco2017.pth
```
Then train a model:
```shell
python tools/train.py \
    configs/EnDiff_r50_diff.py \
    --cfg-options model.init_cfg=None
```

### Generating
As above we make use of `EnDiff-r50` to generate the dataset.
```shell
python tools/generate.py 
```


The results will be saved in `work_dirs/EnDiff_r50/`. 

### Training on a custom dataset
Please convert the annotations into COCO format and place them and images into `data/` folder accoriding to the structure described [above](#data-preperation).

Then, make a copy of the configuration file, and modify following settings:
- `num_classes`: the number of classes.
- `data_root`: the path of the dataset folder.
- `train_ann`: the path of the training annotations.
- `test_ann`: the path of the testing annotations.
- `classes`: a tuple of class names.

Finally, train a model:
```shell
python tools/train.py \
    YOUR_CONFIG_FILE.py
```

The results will be saved in `work_dirs/YOUR_CONFIG_FILE/`.

**Notes:**
- For more information (e.g., about modifying runtime settings), please refer to [MMDetection's documentation](https://mmdetection.readthedocs.io/en/v2.28.2/).

## Citation
```
```