# POPAR: Patch Order Prediction and Appearance Recovery for Self-supervised Medical Image Analysis
This repository provides a PyTorch implementation of the POPAR: Patch Order Prediction and Appearance Recovery for Self-supervised Medical Image Analysis.

We propose POPAR (patch order prediction and appearance recovery), a novel vision transformer-based self-supervised learning framework for chest X-ray images. POPAR leverages the benefits of vision transformers and unique properties of medical imaging, aiming to simultaneously learn patch-wise high-level contextual features by correcting shuffled patch orders and fine-grained features by recovering patch appearance.


<p align="center"><img width="100%" src="images/main.gif" /></p>

## Publication
<b>POPAR: Patch Order Prediction and Appearance Recovery for Self-supervised Medical Image Analysis </b> <br/>
[Jiaxuan Pang](https://github.com/MRJasonP)<sup>1</sup>, [Fatemeh Haghighi](https://github.com/fhaghighi)<sup>1</sup>,[DongAo Ma](https://github.com/Mda233)<sup>1</sup>,[Nahid Ul Islam](https://github.com/Nahid1992)<sup>1</sup>,[Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>
Published in: **Domain Adaptation and Representation Transfer (DART), 2022.**

[Paper](#) | [Code](https://github.com/jlianglab/POPAR) | [Poster](#) | [Slides](#) | Presentation ([YouTube](#), [YouKu](#))



## Major results from our work
1. **POPAR consistently outperforms all state-of-the-art transformer-based self-supervised imagenet pretrained models that are publicly available.**
<br/>
<p align="center"><img width="100%" src="images/result1.png" /></p>
<br/>

2. **Our downgraded POPAR<sup>-1</sup> and POPAR<sup>-3</sup> outperform or achieve on-par performance on most target tasks comapring with all state-of-the-art transformer-based self-supervised imagenet pretrained models that are publicly available.**
<br/>
<p align="center"><img width="100%" src="images/result2.png" /></p>
<br/>

3. **POPAR with Swin-base backbone, even our downgraded version yields significantly better or on-par performance compared with three self-supervised learning methods with ResNet-50 backbone in all target tasks.**
<br/>
<p align="center"><img width="100%" src="images/result3.png" /></p>
<br/>

4. **POPAR models outperform SimMIM in all target tasks across ViT-base and Swin-base backbones.**
<br/>
<p align="center"><img width="100%" src="images/result4.png" /></p>
<br/>

5. **POPAR models outperform fully supervised pretrained models on ImageNet and ChestX-ray14 datasets across architectures**
<br/>
<p align="center"><img width="100%" src="images/result5.png" /></p>
<br/>


Code and pretrained models will be made avalable soon.

## License

Released under the [ASU GitHub Project License](./LICENSE).
