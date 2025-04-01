<div align="center">

  <h1 align="center">HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment</h1>
<!--   <h2 align="center">ICML 2024</h2> -->
  <div align="center">  <img src='static/images/teaser.png' style="height:200px"></img>  </div>

  
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

[Zhichao Liao](https://scholar.google.com/citations?user=4eRwbOEAAAAJ&hl=zh-CN&authuser=1)1 ♰, [Xiaokun Liu]()2, [Wenyu Qin]()2, [Qingyu Li]()2, 
[Qiulin Wang]()2, [Pengfei Wan](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en)2, [Di Zhang]()2, [Long Zeng](https://scholar.google.com/citations?user=72QbaQwAAAAJ&hl=en)1 ✉, [Pingfa Feng]()1 </br>
  
1 Tsinghua University  2 Kuaishou Technology

♰ Internship at KwaiVGI, Kuaishou Technology  ✉ Corresponding Author
  </div>



## 📃 Abstract
Image Aesthetic Assessment (IAA) is a long-standing and challenging research task. However, its subset, Human Image Aesthetic Assessment (HIAA), has been scarcely explored, even though HIAA is widely used in social media, AI workflows, and related domains. To bridge this research gap, our work pioneers a holistic implementation framework tailored for HIAA. Specifically, we introduce HumanBeauty, the first dataset purpose-built for HIAA, which comprises 108k high-quality human images with manual annotations. To achieve comprehensive and fine-grained HIAA, 50K human images are manually collected through a rigorous curation process and annotated leveraging our trailblazing 12-dimensional aesthetic standard, while the remaining 58K with overall aesthetic labels are systematically filtered from public datasets. Based on the HumanBeauty database, we propose HumanAesExpert, a powerful Vision Language Model for aesthetic evaluation of human images. We innovatively design an Expert head to incorporate human knowledge of aesthetic sub-dimensions while jointly utilizing the Language Modeling (LM) and Regression head. This approach empowers our model to achieve superior proficiency in both overall and fine-grained HIAA. Furthermore, we introduce a MetaVoter, which aggregates scores from all three heads, to effectively balance the capabilities of each head, thereby realizing improved assessment precision. Extensive experiments demonstrate that our HumanAesExpert models deliver significantly better performance in HIAA than other state-of-the-art models. Our datasets, models, and codes are publicly released to advance the HIAA community. 





## 🎨 Updates
  - **`2025/04/01`**: Our [**HumanAesExpert paper**](https://arxiv.org/abs/2503.23907) is available.

## 🌏 Open Source
Thank you all for your attention! We are actively cleaning our datasets, models, and codes, and we will open source them soon.


    
## 🖊 Citation
If you find HumanAesExpert useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:

```
@misc{liao2025humanaesexpertadvancingmultimodalityfoundation,
      title={HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment}, 
      author={Zhichao Liao and Xiaokun Liu and Wenyu Qin and Qingyu Li and Qiulin Wang and Pengfei Wan and Di Zhang and Long Zeng and Pingfa Feng},
      year={2025},
      eprint={2503.23907},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.23907}, 
}
```
