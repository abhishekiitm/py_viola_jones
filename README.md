# Viola Jones Cascade Classifier Training
This repository provides a Python implementation for training a cascade classifier using Viola-Jones algorithm. The code has been optimized to run almost as fast as the C++ implementation provided by OpenCV.

## Data preparation
Positive data samples should be placed in a directory
Background images from which negative data will be generated should be placed in another directory.

## Usage
### Training
```
python src/traincascade.py -model /path/to/save/cascade_classifier/ \
 -data_pos /path/to/positive/samples/folder/ \
-data_neg /path/to/background/samples/folder/ \
-numPos [NUM POS] -numNeg [NUM POS] -numStages [NUM STAGES] -maxWeakCount [MAX WEAK CLF COUNT] -minHitRate [MIN DETR EACH ROUND] -maxFalseAlarmRate [MAX FPR EACH ROUND] -log_file /path/to/logfile.log
```

### Detection
```
python src/inference.py -model /path/to/cascade_classifier/ -image /path/to/image.jpg -stage [STAGE] -min_obj_size [MIN OBJ SIZE] -max_obj_size [MAX OBJ SIZE] 
```