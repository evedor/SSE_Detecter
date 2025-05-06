<div align="center">
<h1>SSE_Detecter</h1>
  This is the test repository for the paper 
"Detecting slow slip events in the Cascadia subduction zone from GNSS time series using deep learning".  
</div>

## How to use
1. Clone this repository
2. Install the required packages
3. Place the three-component GNSS time series of single station into the ./Data1 directory.
4. The GNSS time series in ./Data_fulltime will be used for the detection results of the mask (for specific usage and methods, refer to train.ipynb).
5. The model and test code can be found in the train.ipynb file (there is no need to run the training and testing sections).


## Required packages
- numpy
- pandas
- matplotlib
- pytorch
- scikit-learn

## Citation
If you found our project helpful, please kindly cite our paper:

Wang J, Chen K, Zhu H, et al. Detecting slow slip events in the Cascadia subduction zone from GNSS time series using deep learning[J]. GPS Solutions, 2024, 28(4): 156.
```
@article{wang2024detecting,
  title={Detecting slow slip events in the Cascadia subduction zone from GNSS time series using deep learning},
  author={Wang, Ji and Chen, Kejie and Zhu, Hai and Hu, Shunqiang and Wei, Guoguang and Cui, Wenfeng and Xia, Lei},
  journal={GPS Solutions},
  volume={28},
  number={4},
  pages={156},
  year={2024},
  publisher={Springer}
}
```