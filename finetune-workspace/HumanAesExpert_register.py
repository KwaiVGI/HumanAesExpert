from swift.llm.utils.template import *
from swift.llm.utils.model import register_model, LoRATM, _use_submodel_func, BitsAndBytesConfig, get_model_tokenizer_from_repo, _clone_hook
from transformers import AutoTokenizer, AutoConfig

def _findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
    """Find the index of a token in the token_list."""
    if isinstance(sub_token_list, int):
        sub_token_list = [sub_token_list]
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(sub_token_list[0], idx + 1)
            if len(sub_token_list) == 1 or sub_token_list == token_list[idx:idx + len(sub_token_list)]:
                res.append(idx)
    except ValueError:
        pass
    return res

import random
humanaes_expert_prompts = {int(0) : ['Can you evaluate the aesthetics of the human image?', \
                     'Rate the aesthetics of this human picture.', \
                     'How is the aesthetics of this human image?', \
                     'Can you rate the aesthetics of this human picture?', \
                     'Please evaluate the aesthetics of the human image.'],
                    int(1) : ['Can you evaluate the aesthetics of the human image from 12 different dimensions?', \
                                'Rate the aesthetics of this human picture from 12 different dimensions.', \
                                'How is the aesthetics of this human image from 12 different dimensions?', \
                                'Can you rate the aesthetics of this human picture from 12 different dimensions?', \
                                'Please evaluate the aesthetics of the human image from 12 different dimensions.']}

class Internvl2_HumanAesExpertTemplate(InternvlTemplate):    
    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        is_expert = example.get('is_expert')
        example['query'] = '<image>\n' + humanaes_expert_prompts[int(is_expert)][random.randint(0,4)]
        inputs, _ = super(InternvlTemplate, self)._encode(example)
        scores = example.get('scores')
        is_expert = example.get('is_expert')
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        labels = inputs.get('labels')
        images = example.get('images')
        if images:
            has_video = bool(example.get('videos'))
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 1 if has_video else 12)
            pixel_values = [transform_image(image, input_size, max_num) for image in images]
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values).to(self.model.dtype)
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'
        added_tokens_len = 0
        for idx, num_patch in zip(idx_list, num_patches):
            img_tokens: List[int] = self.tokenizer.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * num_patch
            input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
            if labels is not None:
                labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                             + 1:]
            added_tokens_len += len(img_tokens) - 1
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        inputs['scores_labels'] = torch.tensor(scores)
        inputs['_data'] = {'input_ids': torch.tensor(input_ids), 'pixel_values': pixel_values}
        inputs['is_expert'] = torch.tensor(is_expert).to(dtype=torch.int16)
        inputs.pop('loss_scale', None)
        return inputs, {}

class CustomModelType:
    internvl2_1b_HumanAesExpert = 'internvl2_1b_HumanAesExpert'
    internvl2_8b_HumanAesExpert = 'internvl2_8b_HumanAesExpert'


class CustomTemplateType:
    internvl2_HumanAesExpert = 'internvl2_HumanAesExpert'

register_template(CustomTemplateType.internvl2_HumanAesExpert, Internvl2_HumanAesExpertTemplate(), use_model=True, lazy_tokenize=True)

@register_model(
    CustomModelType.internvl2_1b_HumanAesExpert,
    '../../Models/HumanAesExpert-1B',
    LoRATM.internvl,
    CustomTemplateType.internvl2_HumanAesExpert,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='../../Models/HumanAesExpert-1B')
@register_model(
    CustomModelType.internvl2_8b_HumanAesExpert,
    '../../Models/HumanAesExpert-8B',
    LoRATM.internvl,
    CustomTemplateType.internvl2_HumanAesExpert,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='../../Models/HumanAesExpert-8B')
def get_model_tokenizer_internvl_regression(model_dir: str,
                                 torch_dtype: torch.dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    if kwargs.get('eos_token') is None and tokenizer.eos_token != '<|im_end|>':
        try:
            del tokenizer.__class__.eos_token_id
        except AttributeError:
            pass
        tokenizer.eos_token = '<|im_end|>'

    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if hasattr(model_config.llm_config, 'attn_implementation'):
        attr = 'attn_implementation'
    else:
        attr = '_attn_implementation'
    if use_flash_attn:
        setattr(model_config.llm_config, attr, 'flash_attention_2')
    else:
        setattr(model_config.llm_config, attr, 'eager')
        setattr(model_config.llm_config, f'{attr}_internal', None)

    model_quant_config = getattr(model_config, 'quantization_config', None)

    use_bnb = False
    if model_quant_config is not None:
        use_bnb = model_quant_config.get('quant_method', None) == 'bitsandbytes'
    quantization_config = model_kwargs.get('quantization_config', None)
    if isinstance(quantization_config, BitsAndBytesConfig):
        use_bnb = True

    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, tokenizer=tokenizer, model_config=model_config, **kwargs)

    if use_bnb and kwargs.get('is_training'):
        # patch: bnb backward shape mismatch bug
        if model is not None and model.language_model is not None:
            model.language_model.output.state.force_no_igemmlt = True

    if model is not None:
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
        _use_submodel_func(model, 'language_model', func_list)
        embedding = model.language_model.get_input_embeddings()
        embedding.register_forward_hook(_clone_hook)
    
    print('checking is there NaN?\n')
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"param {name} include NaN ")
    print('\n')
    return model, tokenizer

