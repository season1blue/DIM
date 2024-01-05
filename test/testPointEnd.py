# import torch
# from PIL import Image
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor

# image_list = [
#     Image.open("3142.jpg").convert('RGB'),
#     Image.open("509.jpeg").convert('RGB'),
#     Image.open("test2.png").convert('RGB')
# ]

# processor = AutoProcessor.from_pretrained("../../data/pretrain_models/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("../../data/pretrain_models/blip2-opt-2.7b")
# model.to(device)

# question_list = [
#     "Question:  Who is he? Answer: ", 
#     "Question: What is the detail in picture? Answer: "
#     ]

# for raw_image in image_list:
#     inputs = processor(raw_image, return_tensors="pt").to(device, torch.float16)

#     generated_ids = model.generate(**inputs, max_new_tokens=20)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     print(generated_text)


import torch
from PIL import Image
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor

image_list = [
    Image.open("3142.jpg").convert('RGB'),
    Image.open("509.jpeg").convert('RGB'),
    Image.open("test2.png").convert('RGB')
]

processor = AutoProcessor.from_pretrained("../../data/pretrain_models/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("../../data/pretrain_models/blip2-opt-2.7b")
model.to(device)

question = "Question:  Who are the characters in the picture? Answer: "

for raw_image in image_list:
    inputs = processor(raw_image, text=question, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)


