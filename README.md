
# RRInf: Efficient Influence Function Estimation via Ridge Regression for Large Language Models and Text-to-Image Diffusion Models

Official implementation of the paper:  

**RRInf: Efficient Influence Function Estimation via Ridge Regression for Large Language Models and Text-to-Image Diffusion Models**  
*Zhuozhuo Tu, Cheng Chen\, Yuxuan Du*  
📍 Published at **EMNLP 2025 (Main Conference)**  

---

## 📖 Overview

The quality of data plays a vital role in the development of Large-scale Generative Models. Understanding how important a data point is for a generative model is essential for explaining its behavior and improving the performance. The influence function provides a framework for quantifying the impact of individual training data on model predictions. However, the high computational cost has hindered their applicability in large-scale applications. In this work, we present RRInf, a novel and principled method for estimating influence function in large-scale generative AI models. We show that influence function estimation can be transformed into a ridge regression problem. Based on this insight, we develop an algorithm that is efficient and scalable to large models. Experiments on noisy data detection and influential data identification tasks demonstrate that RRInf outperforms existing methods in terms of both efficiency and effectiveness for commonly used large models: RoBERTa-large, Llama-2-13B-chat, Llama-3-8B and stable-diffusion-v1.5.

---

## 🚀 Key Features

- **RRInf Influence Estimation**  
  - Ridge Regression formulation for influence function  
  - Scalable to large-scale generative models  

- **Applications**  
  - **LLMs (e.g., Llama-2, RoBERTa):** SVAMP, MRPC  
  - **Diffusion Models (Stable Diffusion):** Text-to-image style generation 

- **Use Cases**  
  - Identifying **influential training examples**  
  - Detecting **mislabeled / noisy data**  

---

## ⚡ Quick start

Install the required packages:
```bash
pip install -r requirements.txt
```

The Jupyter notebooks in `notebooks` directory demonstrate how to compute influence function and how to detect mislabeled data points and identify most influential training data using computed influence function values:
- Mislabeled Data Detection with RoBERTa model at `notebooks/Mislabeled_Data_Detection-RoBERTa-MRPC.ipynb`.
- Influential Data Identification on text generation with Llama-2 at `notebooks/Influential_Data_Identification-Llama2-SVAMP.ipynb`.
- Influential Data Identification on text-to-image generation with stable diffusion at `notebooks/Influential_Data_Identification-Stable_Diffusion-Style_Generation.ipynb`.

The implementation of RRInf and baseline methods can be found at `src/influence.py`. In all experiments, we first fine-tune a model and then compute influence function. The files for fine-tuning models are given in the `src` directory as follows:
- `lora_model.py` trains a LoRA model and computes first-order gradient for influence function estimation.
- `sft_trainer.py` fine-tunes a llama-2 model. 
- `train_text_to_image_lora.py` fine-tunes a stable diffusion model. 

---

## 📊 Datasets

- **MRPC** — Microsoft Research Paraphrase Corpus  
  - A noisy GLUE-MRPC dataset is generated for mislabeled data detection using `src/dataloader.py`.
- **SVAMP** — Arithmetic math word problem dataset at the `datasets` folder
- **three_styles_prompted_250_512x512** —  Three different styles of image-text datasets (cartoon, sketch, and pixel-art)


## 📒 Reference

```bibtex
@inproceedings{
tu2025rrinf,
title={{RRI}nf: Efficient Influence Function Estimation via Ridge Regression for Large Language Models and Text-to-Image Diffusion Models},
author={Zhuozhuo Tu and Cheng Chen and Yuxuan Du},
booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025}
}
```
---

## 🔗 Acknowledgements

This repository builds upon the [DataInf](https://github.com/ykwon0407/DataInf) project:  
**DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models**.  

We thank the authors of DataInf for providing a strong foundation for our implementation.

---
