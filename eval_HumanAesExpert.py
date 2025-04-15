import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.calculate_metrics import calculate_plcc, calculate_srcc, calculate_krcc, calculate_rmse, calculate_rmae
from utils.judge_answer import map_quality, map_to_id
from dataset_config import HumanBeauty_root, test_data_jsonl
from tqdm import tqdm
import webdataset as wds
import json


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

if __name__ == '__main__':
    # load testset
    with open(test_data_jsonl, 'r') as file:
        lines = file.readlines()

    # load model
    model_path = '../Models/HumanAesExpert-1B'
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # test
    pred_ids, gt_ids, pred_scores, gt_scores = [], [], [], []
    for i, item in tqdm(enumerate(lines), total=12604):
        item = json.loads(item)
        if item['is_expert'] == False:
            gt_score = item['score']
            gt = item['text_level']
        else:
            gt_score = item['score'][-1]
            gt = map_quality(gt_score)

        # pred
        pixel_values = load_image(os.path.join(HumanBeauty_root, item['image_name']), max_num=12).to(torch.float16).cuda()
        pred_score = model.run_metavoter(tokenizer, pixel_values)
        pred_text_level = map_quality(pred_score)
        pred_id, gt_id = map_to_id(pred_text_level, gt)

        pred_scores.append(pred_score)
        gt_scores.append(gt_score)
        pred_ids.append(pred_id)
        gt_ids.append(gt_id)

    # calculate_metrics
    pred_ids, gt_ids = np.array(pred_ids), np.array(gt_ids)
    accuracy = accuracy_score(gt_ids, pred_ids)
    precision = precision_score(gt_ids, pred_ids, average=None)  
    recall = recall_score(gt_ids, pred_ids, average=None)  
    f1 = f1_score(gt_ids, pred_ids, average=None)  
    plcc_score = abs(calculate_plcc(pred_scores, gt_scores))
    srcc_score = abs(calculate_srcc(pred_scores, gt_scores))
    krcc_score = abs(calculate_krcc(pred_scores, gt_scores))
    mse = calculate_rmse(np.array(pred_scores), np.array(gt_scores))
    mae = calculate_rmae(np.array(pred_scores), np.array(gt_scores))

    # save results
    os.makedirs('./results', exist_ok=True)
    output_csv = os.path.join('./results', model_path.split('/')[-1] + '.csv')
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'Accuracy', 'mean Precision', 'mean Recall', 'mean F1', 'plcc', 'srcc', 'krcc', 'mse', 'mae'])
        writer.writerow([model_path, 
                         round(accuracy, 4), 
                         round(np.mean(precision), 4), 
                         round(np.mean(recall), 4), 
                         round(np.mean(f1), 4),
                         round(plcc_score, 4),
                         round(srcc_score, 4),
                         round(krcc_score, 4),
                         round(mse, 4),
                         round(mae, 4)])