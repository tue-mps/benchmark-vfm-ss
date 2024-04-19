# Code for ["How to Benchmark Vision Foundation Models for Semantic Segmentation?"](https://tue-mps.github.io/benchmark-vfm-ss/) (CVPR 2024 Second Workshop on Foundation Models)
<img width="400" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/365962f3-5948-40d7-ae90-256c863ae56c">

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
<img width="400" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/bfc289da-5572-4923-96cb-789ae2dd2dd4"><br>
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
<img width="400" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/c3523c81-27c1-40fd-a850-adde67367baa"><br>
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
<img width="400" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/6d54d1cc-2bc3-49f0-b8de-0398d1661b36"><br>
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
<img width="400" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/c4f8a01e-766e-45a2-9647-d1e50f4fe424"><br>
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
<img width="400" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/4314f4e0-1d0b-4c16-b7fa-322fc398c871"><br>
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
<img width="400" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/9ea2035a-707e-4284-bcb9-14dc5c96a9c4"><br>
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
<img width="400" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/4b4295d6-6aae-4bba-a70a-09f133ff871d"><br>
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
<img width="400" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/e6dfaf1f-b37e-4c09-aeef-7df0fde9bfd2"><br>
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

## Acknowledgement
We borrow some code from Hugging Face Transformers (https://github.com/huggingface/transformers) (Apache-2.0 License)
