# POPAR: Patch Order Prediction and Appearance Recovery for self-supervised learning in chest radiography

This repository provides a PyTorch implementation of the POPAR: Patch Order Prediction and Appearance Recovery for self-supervised learning in chest radiography.

We propose POPAR (patch order prediction and appearance recovery), a novel vision transformer-based self-supervised learning framework for chest X-ray images. POPAR leverages the benefits of vision transformers and unique properties of medical imaging, aiming to simultaneously learn patch-wise high-level contextual features by correcting shuffled patch orders and fine-grained features by recovering patch appearance.

<p align="center"><img width="100%" src="../POPAR_MICCAI2022/images/main.gif" /></p>



<b>POPAR: Patch Order Prediction and Appearance Recovery for self-supervised learning in chest radiography </b> <br/>
[Jiaxuan Pang](https://github.com/MRJasonP)<sup>1</sup>, [DongAo Ma](https://github.com/Mda233)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>
Published in: **Medical Image Analysis, 2025.**

[Paper](https://www.sciencedirect.com/science/article/pii/S1361841525002671) |  [Code](https://github.com/jlianglab/POPAR/tree/main/POPAR/Pretraining) | [Poster] | [Slides] | Presentation ([YouTube])



## Code
### Requirements
+ Python
+ PyTorch ([pytorch.org](http://pytorch.org))
### Setup environment 
Create and activate a Python 3 conda environment:
```
$ conda create -n popar python=3
$ conda activate popar
```
Install PyTorch according to the [CUDA version](https://pytorch.org/get-started/previous-versions/) (e.g., CUDA 11.6)
```
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

$ cd pretraining/
$ pip install -r requirements
```

### Train POPAR model
```
cd Pretraining/

# Train POPAR with NIH ChestX-ray14 on Swin Transformer
python popar_swin.py


# Train POPAR with NIH ChestX-ray14 on Swin Transformer V2
python popar_swinV2.py


# Train POPAR with 811K X-rays on Swin Transformer
python popar_swin.py --dataset allxrays


# Train POPAR with 811K X-rays on Swin Transformer V2
python popar_swinV2.py --dataset allxrays
```



### Finetune the model on target tasks

```
cd Downstream/

# Finetune POPAR pretrained on NIH ChestX-ray14 with Swin Transformer for NIH ChestX-ray14
python nih14_full_tuning.py --weight popar_swin_allxrays_448


# Finetune POPAR pretrained on 811K X-rays with Swin Transformer for JSRT<sub>Lung</sub>
python jsrt_lung.py --weight popar_swin_allxrays_448


# Finetune POPAR pretrained on 811K X-rays with Swin TransformerV2 for VinDr-RibCXR
python vindr_ribcxr.py --weight popar_swinv2_allxrays_512
```


## Models

Our pre-trained Swin Transformer and Swin Transformer V2 models can be downloaded as follows:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Input Resolution (Shuffled Patches)</th>
<th valign="bottom">Pretraining Data</th>
<th valign="bottom">AUC on ChestX-ray14</th>
<th valign="bottom">AUC on CheXpert</th>
<th valign="bottom">AUC on ShenZhen</th>
<th valign="bottom">ACC on RSNA Pneumonia</th>
<th valign="bottom">DICE on NIH Montgomery</th>
<th valign="bottom">DICE on JSRT<sub>Lung</sub></th>
<th valign="bottom">DICE on JSRT<sub>Heart</sub></th>
<th valign="bottom">DICE on JSRT<sub>Clavicle</sub></th>
<th valign="bottom">DICE on VinDr-RibCXR</th>

<th valign="bottom">Model</th>
<!-- TABLE BODY -->

<tr><td align="left">POPAR<sup>Swin</sup><sub>NIH</sub></td>
<td align="center">Swin-B</td>
<td align="center">448x448 (196)</td>
<td align="center">NIH ChestX-ray14</td>
<td align="center">81.81±0.10</td>
<td align="center">88.34±0.50</td>
<td align="center">97.33±0.74</td>
<td align="center">74.19±0.37</td>
<td align="center">98.08±0.11</td>
<td align="center">97.88±0.12</td>
<td align="center">94.47±0.26</td>
<td align="center">91.78±0.53</td>
<td align="center">70.00±0.28</td>
<td align="center"> <a href='https://zenodo.org/records/17356352/files/popar_swin_nih14_448.pth?download=1' target="_blank"> download </a> </td>
</tr>



<tr><td align="left">POPAR<sup>Swin</sup></td>
<td align="center">Swin-B</td>
<td align="center">448x448 (196)</td>
<td align="center">811K X-rays</td>
<td align="center">82.41±0.18</td>
<td align="center"><strong>88.83±0.49</strong></td>
<td align="center">98.03±0.60</td>
<td align="center"><strong>74.72±0.33</strong></td>
<td align="center">98.14±0.04</td>
<td align="center">98.05±0.07</td>
<td align="center">94.80±0.15</td>
<td align="center">92.84±0.50</td>
<td align="center">72.45±0.17</td>
<td align="center"> <a href='https://zenodo.org/records/17356352/files/popar_swin_allxrays_448.pth?download=1' target="_blank"> download </a> </td>
</tr>

<tr><td align="left">POPAR<sub>NIH</sub></td>
<td align="center">SwinV2-B</td>
<td align="center">512x512 (256)</td>
<td align="center">NIH ChestX-ray14</td>
<td align="center">82.67±0.67</td>
<td align="center">88.49±0.54</td>
<td align="center">97.22±0.65</td>
<td align="center">74.06±1.29</td>
<td align="center">98.26±0.21</td>
<td align="center">98.06±0.82</td>
<td align="center">94.62±0.35</td>
<td align="center">91.53±0.44</td>
<td align="center">69.34±1.06</td>
<td align="center"> <a href='https://zenodo.org/records/17356352/files/popar_swinv2_nih14_448.pth?download=1' target="_blank"> download </a> </td>
</tr>


<tr><td align="left">POPAR</td>
<td align="center">SwinV2-B</td>
<td align="center">512x512 (256)</td>
<td align="center">811K X-rays</td>
<td align="center"><strong>83.21±0.37</strong></td>
<td align="center">88.58±0.24 </td>
<td align="center"><strong>98.42±0.45</strong></td>
<td align="center">74.43±0.32</td>
<td align="center"><strong>98.62±0.65</strong></td>
<td align="center"><strong>98.94±0.26</strong></td>
<td align="center"><strong>95.29±0.29</strong></td>
<td align="center"><strong>93.33±0.53</strong></td>
<td align="center"><strong>72.94±1.01</strong></td>
<td align="center"> <a href='https://zenodo.org/records/17356352/files/popar_swinv2_allxrays_512.pth?download=1' target="_blank"> download </a> </td>
</tr>
</tbody></table>

