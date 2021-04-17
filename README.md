# Parallel Res2Net-based Network with Reverse Attention and post-processing for Polyp Segmentation

> **Authors:** 
> [ChengHui Yu](), 
> [JiangPeng Yan](orcid=0000-0002-0767-1726),
> [Xiu Li](orcid=0000-0003-0403-1923),

## 1. Preface
This repository provides code for "_**Parallel Res2Net-based Network with Reverse Attention for Polyp Segmentation**_" EndoCV2021 challenge polyp segmentation task. 

### 3. Experiment
The experiment experiments are conducted using PyTorch with an NVIDIA GeForce RTX 3090 GPU.


1. Configuring your environment (Prerequisites):
   
    Python=3.6, 
    PyTorch=1.2
    
    
1. Training Configuration:
    + Assigning your path, such as `--train_save` and `--train_path` in `Train.py`.

1. Testing Configuration:

    + After you train the model, cd to the `./test/uploas/Test.py`, and run `Test.py` to generate the prediction with post-processing: 
    replace your trained model directory (`--pth_path`).


    
    
   


