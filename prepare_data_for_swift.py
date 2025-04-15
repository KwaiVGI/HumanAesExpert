from dataset_config import HumanBeauty_root, train_data_jsonl
import os
import json
import jsonlines
from utils.judge_answer import map_quality

def arrange(scores_list):
    keys = ['Facial Brightness', 'Facial Feature Clarity', 'Facial Skin Tone', 'Facial Structure', 'Facial Contour Clarity', 'Facial Aesthetic Score', \
                'Outfit', 'Body Shape', 'Looks', 'Environment', 'General Appearance Aesthetic Score']
    text = ""
    for i, key, score in zip(range(11), keys, scores_list[:-1]):
        text += str(i+1) + '. ' + key + ' : ' + map_quality(score) + '\n'

    text += '12. Comprehensive Aesthetic Score : ' + map_quality(scores_list[-1])
    return text

if __name__ == '__main__':
    jsonl_file = os.path.join('./finetune-workspace', 'HumanBeauty-trainset.jsonl')
    with open(train_data_jsonl, 'r') as file:
        lines = file.readlines()

    for line in lines:
        item = json.loads(line)
        image_path = os.path.abspath(os.path.join(HumanBeauty_root, item['image_name']))

        if item['is_expert'] == False:
            score, label = item['score'], item['text_level']
            scores_list = [0.000000 for _ in range(12)]
            scores_list[-1] = float(score)
            data = {'query' : '', # query will be generated during training,
                    'response' : 'The aesthetics of the human image is ' + label,
                    'history' : [],
                    'images' : [image_path],
                    'scores' : scores_list,
                    'is_expert' : int(0)
                    }
        else:
            scores_list = item['score']
            data = {'query' : '', # query will be generated during training,
                    'response' : arrange(scores_list),
                    'history' : [],
                    'images' : [image_path],
                    'scores' : scores_list,
                    'is_expert' : int(1)
                    }

        with jsonlines.open(jsonl_file, mode='a') as writer:
            writer.write(data)