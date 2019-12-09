# TextGraph
Code of the paper:  
# Environment
* python>=3.5
* tensorflow>=1.12.1
* scipy>=1.1.0
# Dataset files
Initial datasets(mr,R52,R8) are from [TextGCN](https://github.com/yao8839836/text_gcn)
* `/data/mr_ori.txt` raw data not processed, Each line is for a document.
* `/data/mr_clear.txt` is the preprocessed data with function `def clean_str()` in `datapro.py`. Each line is for a document.
# Code
To train model 

    python train.py --dataset mr

    

