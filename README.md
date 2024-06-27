 
# DIM:Dynamically Integrate Multimodal Information with Knowledge Base


## :sparkles: Overview

This repository contains official implementation of our paper [DIM:Dynamically Integrate Multimodal Information with Knowledge Base](https://arxiv.org/pdf/2312.11816.pdf).

Our study delves into Multimodal Entity Linking, aligning the mention in multimodal information with entities in knowledge base. Existing methods are still facing challenges like ambiguous entity representations and limited image information utilization. 
Thus, we propose dynamic entity extraction using ChatGPT, which dynamically extracts entities and enhances datasets. We also propose a method, Dynamically Integrate Multimodal information with knowledge base (DIM), employing the capability of the Large Language Model (LLM) for visual understanding. The LLM, such as BLIP2, extracts information relevant to entities in the image, which can facilitate improved extraction of entity features and linking them with the dynamic entity representations provided by ChatGPT.
The experiments demonstrate that our proposed DIM method outperforms the majority of existing methods on the three original datasets, and achieves state-of-the-art (SOTA) on the dynamically enhanced datasets (Wiki+, Rich+, Diverse+).

If you have any question, please feel free to contact me by e-mail: betterszsong@gmail.com or submit your issue in the repository.

## :fire: News

[24.06.27] The paper is accepted by PRCV-24.


## :rocket: Architecture


<p align="center" width="60%"><img src="model.png" alt="DIM" style="width: 100%;  display: block; margin: auto;"></p>


## :rotating_light: Usage
    
### Environment

```
conda create -n DIM python=3.8
conda activate DIM
pip install -r req.txt
```
 

### Data

Original datasets (WikiMEL, Richpeida, Wikidivese) and enhanced datasets (wiki+, rich+, diverse+) could be accessed by [Google Drive](https://drive.google.com/drive/folders/1mFQ2SMZCb3S25E5xsh0rTClHTUo5hkLY?usp=drive_link)

 

### Train
 
The model structure is in nel_model/nel.py, and most of the data processing is in data_process.

You can customize some parameter settings, see nel_model/args.py. Some examples of training are given here:

For training WikiDiverse:
```
sh diverse.sh
```

For training WikiMEL:
```
sh wwiki.sh
```

For training Richpedia:
```
sh rich.sh
```

For training Wikiperson:
```
sh person.sh
```


 

## Citation
```
@misc{song2023dualway,
      title={A Dual-way Enhanced Framework from Text Matching Point of View for Multimodal Entity Linking}, 
      author={Shezheng Song and Shan Zhao and Chengyu Wang and Tianwei Yan and Shasha Li and Xiaoguang Mao and Meng Wang},
      year={2023},
      eprint={2312.11816},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
## License
This repository respects to Apache license 2.0.


