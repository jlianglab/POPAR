<img width="2820" height="919" alt="image" src="https://github.com/user-attachments/assets/061cce3a-d75f-4fd6-8f5e-b8d230deb54e" /># POPAR: Patch Order Prediction and Appearance Recovery for self-supervised learning in chest radiography

This repository provides a PyTorch implementation of the POPAR: Patch Order Prediction and Appearance Recovery for self-supervised learning in chest radiography.

We propose POPAR (patch order prediction and appearance recovery), a novel vision transformer-based self-supervised learning framework for chest X-ray images. POPAR leverages the benefits of vision transformers and unique properties of medical imaging, aiming to simultaneously learn patch-wise high-level contextual features by correcting shuffled patch orders and fine-grained features by recovering patch appearance.

<p align="center"><img width="100%" src="../POPAR_MICCAI2022/images/main.gif" /></p>



<b>POPAR: Patch Order Prediction and Appearance Recovery for self-supervised learning in chest radiography </b> <br/>
[Jiaxuan Pang](https://github.com/MRJasonP)<sup>1</sup>, [DongAo Ma](https://github.com/Mda233)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>
Published in: **Medical Image Analysis, 2025.**

[Paper](https://www.sciencedirect.com/science/article/pii/S1361841525002671) |  [Code](https://github.com/jlianglab/POPAR/tree/main/POPAR/Pretraining) | [Poster] | [Slides] | Presentation ([YouTube])

### Models

Our pre-trained Swin Transformer and Swin Transformer V2 models can be downloaded as following:

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
<td align="center"> <a href='#' target="_blank"> download </a> </td>
</tr>

</tbody></table>

