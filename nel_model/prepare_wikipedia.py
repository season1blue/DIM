import logging
import os
import json
import torch
from PIL import Image
import clip
from tqdm import tqdm

from clip import load as clip_load
from clip import tokenize as clip_tokenize
from pixellib.torchbackend.instance import instanceSegmentation
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor
from args import parse_arg


class InputExample(object):

    def __init__(self, guk, sent, idx, answer=None, mentions=None, img_path=None):
        self.guk = guk  # The unique id of each example, generally composed of mode-key
        self.sent = sent  # Sample text information
        self.img_id = idx  # The original id of the sample, used to retrieve the image
        self.answer = answer  # The answer information corresponding to the sample, that is, the id of the database instance
        self.mentions = mentions  # Reference information in the sample
        self.img_path = img_path


class InputFeatures:
    def __init__(self, answer_id, img_id, mentions, key_id, text_feature, mention_feature, total_feature,
                 segement_feature, caption_feature):
        self.answer_id = answer_id
        self.img_id = img_id
        self.mentions = mentions
        self.key_id = key_id
        self.text_feature = text_feature
        self.mention_feature = mention_feature
        self.total_feature = total_feature
        self.segement_feature = segement_feature
        self.caption_feature = caption_feature


class Wikipedia():
    def __init__(self):
        super(Wikipedia, self).__init__()
        self.args = parse_arg()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        model_path = os.path.join(self.args.pretrain_model_path, "ViT-B-32.pt")
        model, preprocess = clip_load(model_path, device=self.device, jit=False)
        self.model = model
        self.preprocess = preprocess

        self.img_path = self.args.img_path
        Image.MAX_IMAGE_PIXELS = 2300000000  # 更改阈值像素上限

        self.cache_path = self.args.cache_path

        # segment
        self.ins = instanceSegmentation()
        self.ins.load_model(self.args.seg_model_path)
        self.target_classes = self.ins.select_target_classes(person=True)
        self.segement_path = os.path.join(self.args.data_dir, "wiki_segement")
        self.total2part_map = json.load(open(os.path.join(self.args.data_dir, "total2part_map.json"), 'r'))


        self.caption_processor = AutoProcessor.from_pretrained("../../data/pretrain_models/blip2-opt-2.7b")
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained("../../data/pretrain_models/blip2-opt-2.7b")

    def read_examples_from_file(self, data_dir, mode):
        file_path = os.path.join(data_dir, "{}.json".format(mode))
        examples = []

        js = json.load(open(file_path, encoding="utf-8"))

        for k, v in js.items():
            examples.append(
                InputExample(
                    guk=k,  # f"{mode}-{k}",
                    sent=v["sentence"],
                    idx=k,  # v["id"]
                    answer=v["answer"],  # v["answer"]
                    mentions=v["mentions"],
                    img_path=v['imgPath']
                )
            )

        return examples

    # segement image to person image
    def split_image(self, img_path):
        # make sure delete the past segement result
        if len(os.listdir(self.cache_path)) != 0:
            for file in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, file))

        self.ins.segmentImage(img_path, show_bboxes=True, extract_segmented_objects=True,
                              segment_target_classes=self.target_classes, save_extracted_objects=True,
                              output_path=self.cache_path)

        image_list = []
        for file in os.listdir(self.cache_path):
            file_path = os.path.join(self.cache_path, file)
            image = Image.open(file_path)
            image = self.preprocess(image)
            image = image.unsqueeze(0).to(self.device)
            image_list.append(image)
        return image_list

    
    def convert_caption(self, raw_image):
        self.caption_model.to(self.device)

        question = "Question:  Who are the characters in the picture? Answer: "
        inputs = self.caption_processor(raw_image, text=question, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.caption_model.generate(**inputs, max_new_tokens=20)
        generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text



    def convert_examples_to_features(self, examples):
        features = []
        for index, example in tqdm(enumerate(examples), total=len(examples), ncols=80, desc="convert example to features"):
            self.model.to(self.device)
            img_id = example.img_path.split("/")[-1].split(".")[0]
            with torch.no_grad():
                input_sent = example.mentions + " [SEP] " + example.sent
                sent_ids = clip_tokenize(input_sent, truncate=True).to(self.device)  # 截断过长的
                mention = clip_tokenize(example.mentions, truncate=True).to(self.device)

                text_feature = self.model.encode_text(sent_ids)  # text_features 1,512
                mention_feature = self.model.encode_text(mention)

                # extract image feature (split or single)
                img_path = os.path.join(self.img_path, example.img_path)

                total_image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
                total_feature = self.model.encode_image(total_image)

                caption = self.convert_caption(Image.open(img_path))
                print(img_path, caption)
                caption_feature = self.model.encode_text(
                    clip_tokenize(caption, truncate=True).to(self.device)
                )

                if img_id not in self.total2part_map:
                    segement_features = torch.zeros_like(text_feature)
                    # profile_features = torch.zeros_like(text_feature)
                else:
                    segement_list, profile_list = [], []
                    for part in self.total2part_map[img_id]:
                        # segement image feature extraction
                        segement_path = os.path.join(self.segement_path, part + ".jpg")
                        segement = self.preprocess(Image.open(segement_path)).unsqueeze(0).to(self.device)
                        segement_feature = self.model.encode_image(segement)
                        segement_list.append(segement_feature)

                        # # detection profile feature extraction
                        # detection_path = os.path.join(self.detection_path, part + ".json")
                        # detection_context = json.load(open(detection_path, 'r'))[0]
                        # gender, race, age, emotion = detection_context['dominant_gender'], detection_context[
                        #     'dominant_race'], detection_context['age'], detection_context['dominant_emotion']
                        # profile = "gender: {}, race: {}, age: {}, emotion: {}".format(gender, race, age, emotion)
                        # profile = clip_tokenize(profile, truncate=True).to(self.device)
                        # profile_feature = self.model.encode_text(profile)
                        # profile_list.append(profile_feature)

                    segement_features = torch.cat(segement_list, dim=0)
                    # profile_features = torch.cat(profile_list, dim=0)

            features.append(
                InputFeatures(
                    answer_id=example.answer if example.answer else -1,
                    img_id=example.img_id,
                    mentions=example.mentions,
                    key_id=example.guk,
                    text_feature=text_feature,
                    mention_feature=mention_feature,
                    total_feature=total_feature,
                    segement_feature=segement_features,
                    caption_feature=caption_feature,
                )
            )
        return features
