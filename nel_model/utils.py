from transformers import AutoConfig

def build_config():
    text_config =  AutoConfig.from_pretrained("../../data/pretrain_models/text_config.json")
    vision_config =  AutoConfig.from_pretrained("../../data/pretrain_models/vision_config.json")

    return text_config, vision_config   