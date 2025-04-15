import json
import os
import random
from dataset_config import HumanBeauty_root, train_data_jsonl, test_data_jsonl, subpath_58K, subpath_50K

if __name__ == "__main__":

    paths = [subpath_58K] # [subpath_58K, subpath_50K]
    data_by_source = {}
    for path in paths:
        label_file_path = os.path.join(HumanBeauty_root, path, 'label.jsonl')
        with open(label_file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            entry = json.loads(line)
            source = entry['source']
            if source not in data_by_source:
                data_by_source[source] = []
            entry['image_name'] = os.path.join(path, 'images', entry['image_name'])
            if path == subpath_58K:
                entry['is_expert'] = False
            else:
                entry['is_expert'] = True
            data_by_source[source].append(entry)
    train_data = []
    test_data = []
    for source, entries in data_by_source.items():
        random.shuffle(entries)
        split_index = int(len(entries) * 0.9) # an example ratio
        train_data.extend(entries[:split_index])
        test_data.extend(entries[split_index:])

    with open(train_data_jsonl, 'a') as file:
        for entry in train_data:
            file.write(json.dumps(entry) + '\n')

    with open(test_data_jsonl, 'a') as file:
        for entry in test_data:
            file.write(json.dumps(entry) + '\n')
