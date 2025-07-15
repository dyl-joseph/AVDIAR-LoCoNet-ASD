## Modified: LoCoNet: Long-Short Context Network for Active Speaker Detection

Modified LoCoNet for AVDIAR-ASD dataset ([link](https://github.com/UTDTianGroup/AVDIAR2ASD)).

### Dependencies

Start from building the environment
```
conda env create -f environment.yml
conda activate loconet
```
after completing above, manually install the following (using pip)
```
torch 
torchvision
torchaudio 
torchlibrosa 
warmup-scheduler-pytorch
pytorch-lightning
torchmetrics
```
make sure all torch libraries are compatible with CUDA 11.8+ (preferably 12+).

**there might be a package potentially missing from these lists, please install it if you encounter a package/import not found error**


then, run
```
export PYTHONPATH=**project_dir**/dlhammer:$PYTHONPATH
```
and replace **project_dir** with your code base location

### Data preparation

1. Follow the instructions on this GitHub repository: [AVDIAR2ASD](https://github.com/UTDTianGroup/AVDIAR2ASD).
2. Under configs/multi.yaml, modify output directory (line 5) to where you want to store your outputs. Also modify 'dataPathAVA' (line 14) to be the location of your dataset.
3. **IF** the codebase does not auto generate .json files, run
```
cd utils
```
modify variable "phase" to be val or test

then, run:
```
python get_multiperson_csv.py
```

#### Training script
```
python -W ignore::UserWarning train.py --cfg configs/multi.yaml OUTPUT_DIR <output directory>
```
### Model related tasks

#### Pretrained model weights

Please download the LoCoNet trained weights on AVA dataset [here](https://drive.google.com/file/d/1EX-V464jCD6S-wg68yGuAa-UcsMrw8mK/view?usp=sharing).

#### Evaluate model
Run this script to evaluate the model 
```
python -W ignore::UserWarning test_multicard.py --cfg configs/multi.yaml  RESUME_PATH {model download path}
```
07/15/2025: Average Accuracy of 85-86%

#### Get results
To get results,
```
cd visualize_data
bash run.sh
```
the script will run for one video (Seq21-2P-S1M1), and ground truth video and model predicted audio will be generated under visualize_data/outputs. you can modify the variable "video_id" in both python files in visualize_data to get a different scene.

### Citation

Please cite the following (loconet paper): 
```
@article{wang2023loconet,
  title={LoCoNet: Long-Short Context Network for Active Speaker Detection},
  author={Wang, Xizi and Cheng, Feng and Bertasius, Gedas and Crandall, David},
  journal={arXiv preprint arXiv:2301.08237},
  year={2023}
}
```


### Acknowledgement
The code base of this project is from [LoCoNet](https://github.com/SJTUwxz/LoCoNet_ASD). 
The code base of LoCoNet is studied from [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) which is a very easy-to-use ASD pipeline.


