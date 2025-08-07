# Maya: Multimodal Multilingual LLM

- Models and Dataset at [HuggingFace](https://huggingface.co/maya-multimodal)
- Paper: [arXiv](https://arxiv.org/abs/2412.07112)
- Try Maya Model: [Demo](https://huggingface.co/spaces/kkr5155/maya_demo)


<b>Multimodal LLM supporting 8 languages including English, Chinese, French, Spanish, Russian, Japanese, Arabic, Hindi</b>
## Update
- [5/30/2025] Two papers accepted at CVPR Workshops 2025
  - [Understanding and Mitigating Toxicity](https://arxiv.org/abs/2505.06356) as Oral for RGenAI Workshop 2025
  - [Behind Maya](https://arxiv.org/abs/2505.08910) at VLMs4ALL Workshop 2025

## Contents
- [Maya: Multimodal Multilingual LLM](#maya-multimodal-multilingual-llm)
  - [Contents](#contents)
  - [Install](#install)
  - [Model Weights and Dataset](#model-weights-and-dataset)
  - [Train](#train)
    - [Pretraining](#pretraining)
    - [Instruction Tuning](#instruction-tuning)
  - [Evaluation](#evaluation)
  - [Citation](#citation)
  - [Contributors](#contributors)
  - [Acknowledgement](#acknowledgement)

## Install

The following steps worked on a `CUDA Version: 12.4`. 

1. Clone this repository and navigate to maya directory
```bash
git clone https://github.com/nahidalam/maya
cd maya
```

2. Install Package
```Shell
conda create -n maya python=3.10 -y
conda activate maya
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir
```
4. Bump transformers version for Siglip2-Naflex

```
pip install -U transformers==4.50.0
```

## Model Weights and Dataset
[HuggingFace](https://huggingface.co/maya-multimodal)


## Train

### Pretraining

To pretrain the projection layer, 
- get the pretraining dataset from [HuggingFace](https://huggingface.co/maya-multimodal) and keep it in `/dev/data/LLaVA_Pretrain`
- get the images with `wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip` and keep them in `/dev/data/images`
  
```
bash scripts/maya/pretrain_aya_siglip.sh
```

### Instruction Tuning
Please download the annotations from [MBZUAI/palo_multilingual_dataset](https://huggingface.co/datasets/MBZUAI/palo_multilingual_dataset) and all images following the below links.


- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing),
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `/dev/data/instruction_tune_dataset/`,


```
instruction_tune_dataset
    ├── coco
    │   └── train2017
    ├── gqa
    │   └── images
    ├── ocr_vqa
    │   └── images
    ├── textvqa
    │   └── train_images
    └── vg
        ├── VG_100K
        └── VG_100K_2
```

Put the `palo_multilingual_dataset.json` in `/dev/data/annotations/palo_multilingual_dataset.json`

Make sure to keep the pretrained model you have in a path that you specify in the `scripts/maya/finetune_aya_siglip.sh` script throught the `--pretrain_mm_mlp_adapter` flag

Then run
```
bash scripts/maya/finetune_aya_siglip.sh
```

## Evaluation

For multilingual evaluation using PALO multilingual test dataset
- Download the PALO evaluation dataset: Create the following directory structure if it doesn't exist.
  ```
  LLaVA/playground/data/eval
  git clone https://huggingface.co/datasets/MBZUAI/multilingual-llava-bench-in-the-wild
  ```
- Specifically test images can be found [here](https://huggingface.co/datasets/MBZUAI/multilingual-llava-bench-in-the-wild/tree/main/images)
- Run the evaluation script
```
bash scripts/v1_5/eval/eval_all_languages.sh \
    "model_base" \
    "model_path" \
    "model_name" \
    "your-openai-api-key"
```


## Citation

If you find Maya useful for your research and applications, please cite using this BibTeX:
```
@misc{alam2024mayainstructionfinetunedmultilingual,
      title={Maya: An Instruction Finetuned Multilingual Multimodal Model}, 
      author={Nahid Alam and Karthik Reddy Kanjula and Surya Guthikonda and Timothy Chung and Bala Krishna S Vegesna and Abhipsha Das and Anthony Susevski and Ryan Sze-Yin Chan and S M Iftekhar Uddin and Shayekh Bin Islam and Roshan Santhosh and Snegha A and Drishti Sharma and Chen Liu and Isha Chaturvedi and Genta Indra Winata and Ashvanth. S and Snehanshu Mukherjee and Alham Fikri Aji},
      year={2024},
      eprint={2412.07112},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07112}, 
}
```

## Contributors
In no particular order
- Team Leads: [Nahid Alam](https://github.com/nahidalam), [Karthik Reddy](https://github.com/Karthikreddyk99), [Surya Guthikonda](https://github.com/SuryaKrishna02)
- [Timothy Chung](https://github.com/timothycdc)
- [Abhipsha Das](https://github.com/chiral-carbon)
- [Bala Krishna S Vegesna](https://github.com/Satyajitv)
- [Iftekhar Uddin](https://github.com/iuddin)
- [Drishti Sushma](https://github.com/DrishtiShrrrma)
- [Roshan Santhosh](https://github.com/rsk2327)
- [Shayakh Islam](https://github.com/shayekhbinislam)
- [Isha Chaturvedi](https://github.com/ishacusp)
- [Chen Liu](https://github.com/ccliu2)
- [Snegha A](https://github.com/Asnegha)
- [Anthony Susevski](https://github.com/asusevski)
- [Ashvanth.S](https://github.com/ash-01xor)
- [Genta Indra Winata](https://github.com/gentaiscool)
- [Ryan Chan](https://github.com/rchan26)
- [Sangyeon Kim](https://github.com/KimSangYeon-DGU)
- [Snehanshu](https://github.com/pilot-j)


## Acknowledgement

- This codebase is based on [LLaVA](https://github.com/haotian-liu/LLaVA). Thank you for the easily understandable codebase.
- This project would not be possible without the support of Cohere and their Aya-35B API grant. We are thankful to Sara Hooker, Madeline, Shivalika, Shristhi and the entire Cohere for AI team for their support.
- We thank [Merve](https://github.com/merveenoyan) and the HuggingFace team for GPU support for the inference demo
