# TextGraph
Code of the paper:  Modeling Short Texts as Graphs for Classification
# Environment
* python>=3.5
* tensorflow>=1.12.1
* scipy>=1.1.0
# Dataset files
Initial datasets(mr,R52,R8) are from [TextGCN](https://github.com/yao8839836/text_gcn)
* `/data/*_raw.txt` raw data not processed, Each line is for a document.
* `/data/*_clear.txt` is the preprocessed data with function `def clean_str()` in `preData.py`. Each line is for a document.
# Baselines
cited from [Text classification](https://github.com/zhengwsh/text-classification).  
We have made appropriate modifications to fit our data
# Pre-trained word embedding
You can get `glove` form [Glove](https://nlp.stanford.edu/projects/glove/)  
Utilize function `def txt2pkl()` in `preData.py`. 
# Code
To train model 

    python train.py --dataset mr

    

