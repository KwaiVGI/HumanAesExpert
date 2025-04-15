

HumanBeauty_root = '../../../../HumanBeauty' # replace with your path

"""
HumanBeauty
    ---HumanBeauty-58K
        ---images
        ---label.jsonl
    ---HumanBeauty-50K
        ---images
        ---label.jsonl
    ---train_label.jsonl
    ---test_label.jsonl
"""


import os
train_data_jsonl = os.path.join(HumanBeauty_root, 'train_label.jsonl')
test_data_jsonl = os.path.join(HumanBeauty_root, 'test_label.jsonl')
subpath_58K = 'HumanBeauty-58K'
subpath_50K = 'HumanBeauty-50K'