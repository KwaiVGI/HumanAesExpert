import random

mapping_text_to_id = {'excellent' : 0, 'good' : 1, 'fair' : 2, 'poor' : 3, 'bad' : 4}
mapping_id_to_text = {0 : 'excellent', 1 : 'good', 2 : 'fair', 3 : 'poor', 4 : 'bad'}

def map_quality(value):
    if value < 0.200:
        return "bad"
    elif 0.200 <= value < 0.400:
        return "poor"
    elif 0.400 <= value < 0.600:
        return "fair"
    elif 0.600 <= value < 0.800:
        return "good"
    elif 0.800 <= value:
        return "excellent"


def map_to_id(pred, gt):
    pred, gt = pred.lower(), gt.lower()
    for key in mapping_text_to_id.keys():
        if key in pred:
            return mapping_text_to_id[key], mapping_text_to_id[gt]
    # sometime LLMs do not output what we want, random choose one.
    return random.randint(0, 4), mapping_text_to_id[gt]