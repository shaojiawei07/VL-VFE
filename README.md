# VL-VFE
This repository contains the codes for the variable-length variational feature encoding (VL-VFE) method proposed in the [paper](https://arxiv.org/pdf/2102.04170.pdf) "Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck Method", which is accepted to IEEE Journal on Selected Areas in Communications.

## Dependencies
### Packages
```
Pytorch 1.8.1
Torchvision 0.9.1
```
### Datasets
```
MNIST
CIFAR-10
```

## How to run
### Train the VL-VFE method on the MNIST dataset
`python VL_VFE_MNIST.py --intermediate_dim 64  --beta 6e-3 --threshold 1e-2`

### Train the VL-VFE method on the CIFAR dataset
`python VL_VFE_CIFAR.py --intermediate_dim 64  --beta 9e-3 --threshold 1e-2`

The parameter `intermediate_dim` denotes the (maximum) dimension of the encoded feature vector. The weighting factor `beta` and the pruning threshold `threshold` control the tradeoff between the accuracy and the number of activated dimensions.

## Inference
After training the neural network, we can test the performance under different channel conditions `--channel_noise`, which represents the standard deviation in the Gaussian distribution. The relationship between the `--channel_noise` and the peak signal-to-noise ratio (PSNR) is summarized as follows:

| `channel_noise` | 0.3162 |0.2371|0.1778|0.1334|0.1000|0.0750|0.0562|
| :---: | :---: | :---: | :---: |:---: | :---: |:---: | :---: |
|PSNR|10 dB|12.5 dB|15 dB|17.5 dB| 20 dB| 22.5 dB| 25 dB|

### Test the VL-VFE method on the MNIST dataset with PSNR=20 dB

`python3 VL_VFE_MNIST.py --test 1 --intermediate_dim 64 --channel_noise 0.1 --threshold 1e-2 --weights ./pretrained/model/location`

### Test the VL-VFE method on the CIFAR dataset with PSNR=20 dB

`python3 VL_VFE_CIFAR.py --test 1 --intermediate_dim 64 --channel_noise 0.1 --threshold 1e-2 --weights ./pretrained/model/location`

Several pretrained models and results are shown in [Examples](https://github.com/shaojiawei07/VL-VFE/tree/main/Examples).


## Citation

```
@article{shao2021learning,  
  author={Shao, Jiawei and Mao, Yuyi and Zhang, Jun},  
  journal={IEEE Journal on Selected Areas in Communications},  
  title={Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck Approach},   
  year={2022},  
  volume={40},  
  number={1},  
  pages={197-211},  
  doi={10.1109/JSAC.2021.3126087}}
```
## Others

* The variational feature encoding (VFE) proposed in this paper can be achieved by replacing the function `self.gamma_mu = gamma_function()` with a vector `self.mu = nn.Parameter(torch.ones(args.intermediate_dim))` and fixing the channel noise level in the training process.

* Known problem: The loss may become `NaN` when training the network on the CIFAR dataset.

* (2023-Jan-19) I have corrected some typos and updated the paper on [arxiv](https://arxiv.org/pdf/2102.04170.pdf).

