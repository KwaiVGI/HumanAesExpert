<div align="center">

  <h1 align="center">HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment</h1>
<!--   <h2 align="center">ICML 2024</h2> -->
  <div align="center">  <img src='static/images/teaser.png' style="height:250px"></img>  </div>

  
  <div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://arxiv.org/abs/2503.23907'><img src='https://img.shields.io/badge/arXiv-HumanAesExpert-red'></a>  &nbsp;
  <a href='https://humanaesexpert.github.io/HumanAesExpert/'><img src='https://img.shields.io/badge/Project-HumanAesExpert-green'></a> &nbsp;
  <a href="https://github.com/HumanAesExpert/HumanAesExpert"><img src="https://img.shields.io/badge/GitHub-HumanAesExpert-9E95B7?logo=github"></a> &nbsp; 
  <br>
  <a href='https://huggingface.co/HumanBeauty/HumanAesExpert-1B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-HumanAesExpert_1b-blue'></a> &nbsp; 
  <a href='https://huggingface.co/HumanBeauty/HumanAesExpert-8b'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-HumanAesExpert_8b-blue'></a> &nbsp; 
  <a href='https://huggingface.co/datasets/HumanBeauty/HumanBeauty_Dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20%20Dataset-HumanBeauty-blue'></a> &nbsp;
<!--   <a href='https://huggingface.co/spaces/KwaiVGI/VideoGen-RewardBench'><img src='https://img.shields.io/badge/Space-VideoGen--RewardBench-orange.svg?logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzM5MjA0MzY2MDEwIiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjQzNDYiIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48cGF0aCBkPSJNNjgyLjY2NjY2NyA0NjkuMzMzMzMzVjEyOEgzNDEuMzMzMzMzdjI1Nkg4NS4zMzMzMzN2NTEyaDg1My4zMzMzMzRWNDY5LjMzMzMzM2gtMjU2eiBtLTI1Ni0yNTZoMTcwLjY2NjY2NnY1OTcuMzMzMzM0aC0xNzAuNjY2NjY2VjIxMy4zMzMzMzN6IG0tMjU2IDI1NmgxNzAuNjY2NjY2djM0MS4zMzMzMzRIMTcwLjY2NjY2N3YtMzQxLjMzMzMzNHogbTY4Mi42NjY2NjYgMzQxLjMzMzMzNGgtMTcwLjY2NjY2NnYtMjU2aDE3MC42NjY2NjZ2MjU2eiIgcC1pZD0iNDM0NyIgZmlsbD0iIzhhOGE4YSI+PC9wYXRoPjwvc3ZnPg=='></a> &nbsp; -->
  </div>
  <br>
<!--  ### [[`Project Page`](https://humanaesexpert.github.io/HumanAesExpert/)][[`arxiv`](https://arxiv.org/)][[`Paper`](https://arxiv.org/)][[`HumanAesExpert-1B`](https://huggingface.co/HumanBeauty/HumanAesExpert-1B)][[`HumanAesExpert-8B`](https://huggingface.co/HumanBeauty/HumanAesExpert-8B)][[`HumanBeauty`](https://huggingface.co/datasets/HumanBeauty/HumanBeauty-58K)] -->

[**Zhichao Liao**](https://lzc-sg.github.io/)<sup>1</sup> <sup>♰</sup>, [Xiaokun Liu]()<sup>2</sup>, [Wenyu Qin]()<sup>2</sup>, [Qingyu Li]()<sup>2</sup>, 
[Qiulin Wang]()<sup>2</sup>,  
[Pengfei Wan](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en)<sup>2</sup>, [Di Zhang]()<sup>2</sup>, [Long Zeng](https://scholar.google.com/citations?user=72QbaQwAAAAJ&hl=en)<sup>1</sup> <sup>✉</sup>, [Pingfa Feng]()<sup>1</sup>

<sup>1</sup> Tsinghua University,    <sup>2</sup> Kuaishou Technology

♰ Internship at KwaiVGI, Kuaishou Technology  ✉ Corresponding Author
  </div>


## 🎨 News
  - **`2025/04/01`**: Our [**HumanAesExpert paper**](https://arxiv.org/abs/2503.23907) is available.


## 🌏 Open Source
Thank you all for your attention! We are actively cleaning our datasets, models, and codes, and we will open source them soon.
- [x] Technical Paper
- [ ] Training and inference code
- [ ] Pre-trained Models (HumanAesExpert-1B, HumanAesExpert-8B)
- [ ] HumanBeauty Dataset


## 📃 Abstract
Image Aesthetic Assessment (IAA) is a long-standing and challenging research task. However, its subset, Human Image Aesthetic Assessment (HIAA), has been scarcely explored, even though HIAA is widely used in social media, AI workflows, and related domains. To bridge this research gap, our work pioneers a holistic implementation framework tailored for HIAA. Specifically, we introduce <span style="color: red;">_**HumanBeauty**_</span>, the first dataset purpose-built for HIAA, which comprises 108k high-quality human images with manual annotations. To achieve comprehensive and fine-grained HIAA, 50K human images are manually collected through a rigorous curation process and annotated leveraging our trailblazing 12-dimensional aesthetic standard, while the remaining 58K with overall aesthetic labels are systematically filtered from public datasets. Based on the HumanBeauty database, we propose <span style="color: red;">_**HumanAesExpert**_</span> a powerful Vision Language Model for aesthetic evaluation of human images. We innovatively design an Expert head to incorporate human knowledge of aesthetic sub-dimensions while jointly utilizing the Language Modeling (LM) and Regression head. This approach empowers our model to achieve superior proficiency in both overall and fine-grained HIAA. Furthermore, we introduce a MetaVoter, which aggregates scores from all three heads, to effectively balance the capabilities of each head, thereby realizing improved assessment precision. Extensive experiments demonstrate that our HumanAesExpert models deliver significantly better performance in HIAA than other state-of-the-art models. Our datasets, models, and codes are publicly released to advance the HIAA community. 


## 📊 HumanBeauty Dataset Construction Pipeline
<p align="center">
<img src="static/images/data-pipeline.png" width=100% height=100% 
class="center">
</p>

First, we select six diverse open-source datasets as data sources and perform data filtering to build our HumanBeauty-58k. Additionally, we manually collect and annotate 50k human images across multiple dimensions to create our HumanBeauty-50k. Finally, we map all the scores into text of rating level to form QA pairs for training.


## 🧭 HumanAesExpert Overview
<p align="center">
<img src="static/images/model-overview.png" width=100% height=100% 
class="center">
</p>

Our approach uses both LM and Regression heads, along with an Expert head and MetaVoter.

## 🥕 Visualization Results 
<p align="center">
<img src="static/images/cases.png" width=100% height=100% 
class="center">
</p>

The Visualization Results of Our Model, where ‘’( )'' indicate the Ground Truth scores. From A to L, they respectively represent facial brightness, facial feature clarity, facial skin tone, facial structure, facial contour clarity, facial aesthetic, outfit, body shape, looks, general appearance aesthetic, environment and overall aesthetic scores.


    
## 🖊 Citation
If you find HumanAesExpert useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:

```
@article{liao2025humanaesexpert,
  title={HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment},
  author={Liao, Zhichao and Liu, Xiaokun and Qin, Wenyu and Li, Qingyu and Wang, Qiulin and Wan, Pengfei and Zhang, Di and Zeng, Long and Feng, Pingfa},
  journal={arXiv preprint arXiv:2503.23907},
  year={2025}
}
```
