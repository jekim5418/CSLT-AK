# CSLT-AK: Convolution-Embedded Transformer with Action Tokenizer and Keypoint Emphasizer for Sign Language Translation
## Accepted in Pattern recognition letters

[[Paper]](💥)

![1-s2 0-S0167865523002283-gr3_lrg](https://github.com/jekim5418/CSLT-AK/assets/60121575/25db17e2-15fc-4000-9774-9457a63b9c62)

> CSLT-AK: Convolution-Embedded Transformer with Action Tokenizer and Keypoint Emphasizer for Sign Language Translation
>
>Pattern recognition letters 2023\
>Jungeun Kim, Ha Young Kim\
>Yonsei University
>

## Abstract

Sign language translation is a complex task that involves generating spoken-language sentences from sign language (SL) videos, considering the signer's manual and nonmanual movements. We observed the following issues with existing SL translation (SLT) methods and datasets for improving performance. First, every SL video frame does not have gloss notation. Second, nonmanual components can be easily overlooked despite their importance because they occur in small areas of the image. Third, recent SLT models, based on the transformer, have numerous parameters and struggle to capture the local context of SL images comprehensively. To address these problems, we propose an action tokenizer that divides SL videos into semantic units. In addition, we design a keypoint emphasizer and convolutional-embedded SL transformer (CSLT) to understand noticeable manual and subtle nonmanual features effectively. By applying the proposed modules to Sign2 (Gloss+Text), we introduce CSLT with an action tokenizer and keypoint emphasizer (CSLT-AK), a simple yet efficient and effective SLT model based on domain knowledge. The experimental results on the RWTH-PHOENIX-Weather 2014 T reveal that CSLT-AK surpasses the baseline regarding performance and parameter reduction and demonstrates competitive performance without the need for regularization methods compared to other state-of-the-art models.

## Installation

### Environment

Firstly, it is recommanded to download pytorch V1.10.0 corresponding your CUDA from [Pytorch](https://pytorch.org/get-started/previous-versions/). 
```
pip install -r requirements.txt 
```

### Download datasets

Download action-tokens, keypoints, and feature files of Phoenix2014T using data/download.sh script.
```
cd data
sh download.sh
```

## Training
At the project root,
```
python -m signjoey train configs/cslt-ak.yaml
```

### Download best_models for inference

Download trained best models.
```
cd saved_model
sh download.sh
```

## Evaluation
We included the best models of CSLT-AK in 'saved_model' folder.

### Inference
At the project root,
- CSLT-AK(Single model)
```
python -m signjoey test configs/cslt-ak-single.yaml
```

- CSLT-AK-ensemble(Ensemble model)
```
python -m signjoey test configs/cslt-ak-ensemble.yaml
```

If you want to inference your own model, modify 'model_dir' in 'configs/cslt-ak-single.yaml' or 'configs/cslt-ak-ensemble.yaml'.

--
## Citation
If you want to cite the following paper:
'''
@article{KIM2023115,
	title = {CSLT-AK: Convolutional-embedded transformer with an action tokenizer and keypoint emphasizer for sign language translation},
	journal = {Pattern Recognition Letters},
	volume = {173},
	pages = {115-122},
	year = {2023},
	issn = {0167-8655},
	doi = {<https://doi.org/10.1016/j.patrec.2023.08.009>},
	url = {<https://www.sciencedirect.com/science/article/pii/S0167865523002283>},
	author = {Jungeun Kim and Ha Young Kim},
}
'''
