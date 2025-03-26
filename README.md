# Code for ["How to Benchmark Vision Foundation Models for Semantic Segmentation?"](https://tue-mps.github.io/benchmark-vfm-ss/) (CVPR'24 Second Workshop on Foundation Models)

**September 30, 2024** – 🚀 Code used to achieve **1st place** in ECCV'24 BRAVO Challenge ([Submission report](https://arxiv.org/abs/2409.17208), [Workshop paper](https://arxiv.org/abs/2409.15107))

## Getting started
1. **Download datasets.**
    Downloading is optional depending on which datasets you intend to use.

    - **ADE20K**: [Download](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
    - **PASCAL VOC**: [Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    - **Cityscapes**: [Download 1](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [Download 2](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
    - **GTA V**: [Download 1](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip) | [Download 2](https://download.visinf.tu-darmstadt.de/data/from_games/data/02_images.zip) | [Download 3](https://download.visinf.tu-darmstadt.de/data/from_games/data/03_images.zip) | [Download 4](https://download.visinf.tu-darmstadt.de/data/from_games/data/04_images.zip) | [Download 5](https://download.visinf.tu-darmstadt.de/data/from_games/data/05_images.zip) | [Download 6](https://download.visinf.tu-darmstadt.de/data/from_games/data/06_images.zip) | [Download 7](https://download.visinf.tu-darmstadt.de/data/from_games/data/07_images.zip) | [Download 8](https://download.visinf.tu-darmstadt.de/data/from_games/data/08_images.zip) | [Download 9](https://download.visinf.tu-darmstadt.de/data/from_games/data/09_images.zip) | [Download 10](https://download.visinf.tu-darmstadt.de/data/from_games/data/10_images.zip) | [Download 11](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip) | [Download 12](https://download.visinf.tu-darmstadt.de/data/from_games/data/02_labels.zip) | [Download 13](https://download.visinf.tu-darmstadt.de/data/from_games/data/03_labels.zip) | [Download 14](https://download.visinf.tu-darmstadt.de/data/from_games/data/04_labels.zip) | [Download 15](https://download.visinf.tu-darmstadt.de/data/from_games/data/05_labels.zip) | [Download 16](https://download.visinf.tu-darmstadt.de/data/from_games/data/06_labels.zip) | [Download 17](https://download.visinf.tu-darmstadt.de/data/from_games/data/07_labels.zip) | [Download 18](https://download.visinf.tu-darmstadt.de/data/from_games/data/08_labels.zip) | [Download 19](https://download.visinf.tu-darmstadt.de/data/from_games/data/09_labels.zip) | [Download 20](https://download.visinf.tu-darmstadt.de/data/from_games/data/10_labels.zip)

2. **Environment setup.**
    ```bash
    conda create -n benchmark-vfm-ss python=3.10
    conda activate benchmark-vfm-ss
    ```

3. **Install required packages.**
    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu123
    ```
    (replace with your CUDA version if not 12.3).

4. **Fine-tune a model.**
   Here's an example for fine-tuning DINOv2 on ADE20K with the default setup on GPU 0 with 1 worker for data loading:
   ```bash
   python main.py fit -c configs/ade20k_linear_semantic.yaml --root /data --data.num_workers 1 --trainer.devices [0] --model.network.encoder_name vit_base_patch14_dinov2
   ```
   (replace ```/data``` with the folder where you stored the datasets)  

## Reproducing results from the paper
For the commands below, add `--root` to specify the path to where the datasets and checkpoints are stored and `--data.num_workers` to specify the number of workers for data     loading.

If using the BEiT models, download their checkpoints and convert them to timm format using `convert_beit_ckpt.ipynb`.
- **Base**: [Download](https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_224.pth)
- **Large**: [Download](https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_224.pth)

Please note that:
- BEiT models need a checkpoint from above (which is loaded with `--model.network.ckpt_path`) and apply layernorm slightly differently (so the architecture is modified with `--model.network.sub_norm`).
- EVA02 models somehow show significantly lower mIoU when using `torch.compile` (so it is turned off with `--no_compile`).

### Default setup:  
1. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
2. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
3. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
4. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
5. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
6. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
7. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
8. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
9. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
10. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

### Freezing the encoder:  
1. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile --model.freeze_encoder True```  
2. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile --model.freeze_encoder True```  
3. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2 --model.freeze_encoder True```  
4. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True --model.freeze_encoder True```  
5. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli --model.freeze_encoder True```  
6. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b --model.freeze_encoder True```  
7. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k --model.freeze_encoder True```  
8. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k --model.freeze_encoder True```  
9. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae --model.freeze_encoder True```  
10. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b --model.freeze_encoder True```  

### Changing the decoder:  
1. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
2. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```   
3. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
4. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
5. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
6. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
7. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
8. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
9. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
10. ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

### Scaling the model:  
1. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_large_patch14_clip_336.merged2b --no_compile```  
2. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_large_patch14_224.mim_m38m --no_compile```  
3. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch14_dinov2```  
4. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch16_224 --model.network.ckpt_path beit3_large_patch16_224.pth.timm --model.network.sub_norm True```  
5. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch16_siglip_384.webli```  
6. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch14_clip_224.dfn2b```  
7. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_large_patch16_384.fb_in22k_ft_in1k```  
8. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_large_patch16_384.fb_in1k```  
9. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch16_224.mae```  
10. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name samvit_large_patch16.sa1b```  

### Varying the patch size:  
1. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile --model.network.patch_size 8```  
2. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
3. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2  --model.network.patch_size 8```  
4. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True --model.network.patch_size 8```  
5. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli --model.network.patch_size 8```  
6. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b --model.network.patch_size 8```  
7. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k --model.network.patch_size 8```  
8. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k --model.network.patch_size 8```   
9. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae --model.network.patch_size 8```  
10. ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b --model.network.patch_size 8```  

### Changing the downstream dataset (PASCAL VOC):  
1. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
2. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
3. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
4. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
5. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
6. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
7. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
8. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
9. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
10. ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

### Changing the downstream dataset (Cityscapes):  
1. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
2. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
3. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
4. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
5. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
6. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
7. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
8. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
9. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
10. ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

### Introducing a domain shift:  
1. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
2. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
3. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
4. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path - beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
5. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
6. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
7. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
8. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
9. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
10. ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

## 📄 Citation
If you use this code in your research or project, please cite the related paper(s):

```bibtex
@inproceedings{kerssies2024benchmarking,
  author={Kerssies, Tommie and de Geus, Daan and Dubbelman, Gijs},
  title={How to Benchmark Vision Foundation Models for Semantic Segmentation?},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2024},
}
```

```bibtex
@article{kerssies2024evaluating,
  title = {First Place Solution to the ECCV 2024 BRAVO Challenge: Evaluating Robustness of Vision Foundation Models for Semantic Segmentation},
  author = {Kerssies, Tommie and de Geus, Daan and Dubbelman, Gijs},
  journal = {arXiv preprint arXiv:2409.17208},
  year = {2024},
}
```

```bibtex
@inproceedings{vu2024bravo,
  title = {The BRAVO Semantic Segmentation Challenge Results in UNCV2024},
  author = {Vu, Tuan-Hung and Valle, Eduardo and Bursuc, Andrei and Kerssies, Tommie and de Geus, Daan and Dubbelman, Gijs and Qian, Long and Zhu, Bingke and Chen, Yingying and Tang, Ming and Wang, Jinqiao and Vojíř, Tomáš and Šochman, Jan and Matas, Jiří and Smith, Michael and Ferrie, Frank and Basu, Shamik and Sakaridis, Christos and Van Gool, Luc},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV) Workshops},
  year = {2024},
}
```

## Acknowledgement
We borrow some code from Hugging Face Transformers (https://github.com/huggingface/transformers) (Apache-2.0 License)
