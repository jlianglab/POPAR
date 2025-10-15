# POPAR: Patch Order Prediction and Appearance Recovery for Self-supervised Medical Image Analysis
This repository provides a PyTorch implementation of the POPAR: Patch Order Prediction and Appearance Recovery for Self-supervised Medical Image Analysis.

We propose POPAR (patch order prediction and appearance recovery), a novel vision transformer-based self-supervised learning framework for chest X-ray images. POPAR leverages the benefits of vision transformers and unique properties of medical imaging, aiming to simultaneously learn patch-wise high-level contextual features by correcting shuffled patch orders and fine-grained features by recovering patch appearance.

<p align="center"><img width="100%" src="POPAR_MICCAI2022/images/main.gif" /></p>

## Publication
<b>POPAR: Patch Order Prediction and Appearance Recovery for Self-supervised Medical Image Analysis </b> <br/>
[Jiaxuan Pang](https://github.com/MRJasonP)<sup>1</sup>, [Fatemeh Haghighi](https://github.com/fhaghighi)<sup>1</sup>,[DongAo Ma](https://github.com/Mda233)<sup>1</sup>,[Nahid Ul Islam](https://github.com/Nahid1992)<sup>1</sup>,[Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>
Published in: **Domain Adaptation and Representation Transfer (DART), 2022.**

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-16852-9_8) | [Supplementary material](https://link.springer.com/chapter/10.1007/978-3-031-16852-9_8#Sec9) |  [Code](https://github.com/jlianglab/POPAR) | [Poster] | [Slides] | Presentation ([YouTube])


<b>POPAR: Patch Order Prediction and Appearance Recovery for self-supervised learning in chest radiography </b> <br/>
[Jiaxuan Pang](https://github.com/MRJasonP)<sup>1</sup>, [DongAo Ma](https://github.com/Mda233)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>
Published in: **Medical Image Analysis, 2025.**

[Paper](https://www.sciencedirect.com/science/article/pii/S1361841525002671) |  [Code](https://github.com/jlianglab/POPAR/tree/main/POPAR/Pretraining) | [Poster] | [Slides] | Presentation ([YouTube])





## Citation
If you use this code or use our pre-trained weights for your research, please cite our paper:
```
@inproceedings{pang2022popar,
  title={Popar: Patch order prediction and appearance recovery for self-supervised medical image analysis},
  author={Pang, Jiaxuan and Haghighi, Fatemeh and Ma, DongAo and Islam, Nahid Ul and Hosseinzadeh Taher, Mohammad Reza and Gotway, Michael B and Liang, Jianming},
  booktitle={MICCAI Workshop on Domain Adaptation and Representation Transfer},
  pages={77--87},
  year={2022},
  organization={Springer}
}

@article{pang2025popar,
  title={POPAR: Patch Order Prediction and Appearance Recovery for self-supervised learning in chest radiography},
  author={Pang, Jiaxuan and Ma, Dongao and Zhou, Ziyu and Gotway, Michael B and Liang, Jianming},
  journal={Medical image analysis},
  pages={103720},
  year={2025},
  publisher={Elsevier}
}

```

## Acknowledgement
This research has been supported in part by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and in part by the NIH under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided in part by the ASU Research Computing and in part by the Bridges-2 at Pittsburgh Supercomputing Center through allocation BCS190015 and the Anvil at Purdue University through allocation MED220025 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296. We also acknowledge Google for granting us access to CXR Foundation API, which enabled us to generate the embeddings for the target datasets. The content of this paper is covered by patents pending.


## License

Released under the [ASU GitHub Project License](./LICENSE).
