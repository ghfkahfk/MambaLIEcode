# MambaLIE
This repo contains the code for our paper "MambaLIE: Scene Light Intensity-Boosted Low-Light Image Enhancement with State Space Model''
![image](https://github.com/shiyi0306/DGPANet/blob/main/DGPANet_framework.png?raw=true) 

### Dependencies
Please install following essential dependencies:
```
conda create -n LLFormer python=3.7
conda activate LLFormer
conda install pytorch=1.8 torchvision=0.3 cudatoolkit=10.1 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

### Datasets and pre-processing
You can use the following links to download the datasets
Download:  
1. **MIT-Adobe FiveK(code: yvhi)**  [https://pan.baidu.com/share/init?surl=z4sBVXdn8eJv1VpSI0LohA&pwd=yvhi)  
2. **LOL-v1(code: cyh2)** [https://pan.baidu.com/share/init?surl=ZAC9TWR-YeuLIkWs3L7z4g&pwd=cyh2)  
3. **LOL-v2(code: cyh2)** [https://pan.baidu.com/share/init?surl=X4HykuVL_1WyB3LWJJhBQg&pwd=cyh2)  



### Training  
python train.py -yml_path your_config_path

### Testing
# Tesing parameter 
input_dir # the path of data
result_dir # the save path of results 
weights # the weight path of the pre-trained model
python test.py --input_dir your_data_path --result_dir your_save_path --weights weight_path


### Acknowledgement
This code is based on [LLformer](https://doi.org/10.1609/aaai.v37i3.25364) by [Wang et al.](https://github.com/ZJLAB-AMMI/Q-Net).
