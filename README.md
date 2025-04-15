<div align="center">

  <h1 align="center">HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment</h1>
<!--   <h2 align="center">ICML 2024</h2> -->
  <div align="center">  <img src='static/images/teaser.png' style="height:250px"></img>  </div>

  
  <div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://arxiv.org/abs/2503.23907'><img src='https://img.shields.io/badge/arXiv-HumanAesExpert-red'></a>  &nbsp;
  <a href='https://humanaesexpert.github.io/HumanAesExpert/'><img src='https://img.shields.io/badge/Project-HumanAesExpert-green'></a> &nbsp;
  <a href="https://github.com/KwaiVGI/HumanAesExpert"><img src="https://img.shields.io/badge/GitHub-HumanAesExpert-9E95B7?logo=github"></a> &nbsp; 
  <br>
  <a href='https://huggingface.co/KwaiVGI/HumanAesExpert-1B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-HumanAesExpert_1b-blue'></a> &nbsp; 
  <a href='https://huggingface.co/KwaiVGI/HumanAesExpert-8B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-HumanAesExpert_8b-blue'></a> &nbsp; 
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
  - **`2025/04/15`**: We warmly welcome you to try HumanAesExpert! 🥕🥕🥕
  - **`2025/04/15`**: We release the HumanAesExpert-1B and HumanAesExpert-8B pre-trained models on Hugging Face! 🔥🔥🔥 
  - **`2025/04/15`**: We release the [**training and inference code on Github**](https://github.com/KwaiVGI/HumanAesExpert)! 🔥🔥🔥 
  - **`2025/04/01`**: Our [**HumanAesExpert paper**](https://arxiv.org/abs/2503.23907) is available.


## 🌏 Open Source
Thank you all for your attention! We are actively cleaning our datasets, models, and codes, and we will open source them soon.
- [x] Technical Paper
- [x] Training and inference code on GitHub
- [x] Pre-trained models (HumanAesExpert-1B, HumanAesExpert-8B) on Hugging Face
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



## 🔜 Quicker Start with Hugging Face AutoModel

No need to install this GitHub repo. 
> Please use transformers==4.44.2 to ensure the model works normally.


```python
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = 'HumanBeauty/HumanAesExpert-1B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

pixel_values = load_image('./examples/your_image.jpg', max_num=12).to(torch.float16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)

question = '<image>\nRate the aesthetics of this human picture.'

# fast inference, need 1x time
pred_score = model.score(tokenizer, pixel_values, question)

# slow inference, need 2x time
metavoter_score = model.run_metavoter(tokenizer, pixel_values)

# get expert scores from the Expert head, include 12 dimensions
expert_score, expert_text = model.expert_score(tokenizer,pixel_values)

# get expert annotations from the LM head, include 12 dimensions
expert_annotataion = model.expert_annotataion(tokenizer, pixel_values, generation_config)
```

## 🔧 Installation

If you need to fine-tune or train your model from scratch, you need to install additional dependencies further. Our training code is based on modified [ms-swift](https://github.com/modelscope/ms-swift), and you should install it from this repository instead of the official code.

```shell
git clone https://github.com/HumanAesExpert/HumanAesExpert
cd HumanAesExpert
conda create -n HumanAesExpert python=3.10.14 -y
conda activate HumanAesExpert
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd swift
pip install -e .
```


## 📈 Prepare Data
Get dataset ready, you can arrange your dirs as follows:

```
your_root
    ---HumanBeauty
        ---HumanBeauty-58K
            ---images
            ---label.jsonl
        ---HumanBeauty-50K
            ---images
            ---label.jsonl
        ---train_label.jsonl
        ---test_label.jsonl
    ---HumanAesExpert (this repository)
```

or update 'HumanBeauty_root' in [dataset_config.py](dataset_config.py)
We need to produce jsonl file for fine-tuning. The following command will generate a jsonl file in [finetune-workspace](finetune-workspace).
```shell
python prepare_data_for_swift.py
```

## 🔥 Fine-tune HumanAesExpert
```shell
cd finetune-workspace
source ./finetune-HumanAesExpert-1b.sh
```

## 🔥 Fine-tune InternVL2 from scratch
The simplest approach is to download the original weights corresponding to the model size of InternVL2 and replace our weight files. Then, change the model path to the local path.



## 💗 Acknowledgements

We are immensely grateful to the [ms-swift](https://github.com/modelscope/ms-swift) and [InternVL](https://github.com/OpenGVLab/InternVL) projects for the inception of this repository.




## ⚖️ License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.


    
## 🖊 Citation
If you find HumanAesExpert useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:

```bibtex
@article{liao2025humanaesexpert,
  title={HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment},
  author={Liao, Zhichao and Liu, Xiaokun and Qin, Wenyu and Li, Qingyu and Wang, Qiulin and Wan, Pengfei and Zhang, Di and Zeng, Long and Feng, Pingfa},
  journal={arXiv preprint arXiv:2503.23907},
  year={2025}
}
```
