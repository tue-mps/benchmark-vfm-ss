# Code for ["How to Benchmark Vision Foundation Models for Semantic Segmentation?"](https://arxiv.org/abs/2404.12172)
<img width="500" alt="image" src="https://github.com/tue-mps/benchmark-vfm-ss/assets/6392002/5a917336-1205-4e19-a74c-efb36e4cfa20">

## Getting started
1. **Download datasets**:
    Downloading is optional depending on which datasets you intend to use.

    - **ADE20K**: [Download](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
    - **PASCAL VOC**: [Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    - **Cityscapes**: [Download 1](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [Download 2](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
    - **GTA V**: [Download 1](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip) | [Download 2](https://download.visinf.tu-darmstadt.de/data/from_games/data/02_images.zip) | [Download 3](https://download.visinf.tu-darmstadt.de/data/from_games/data/03_images.zip) | [Download 4](https://download.visinf.tu-darmstadt.de/data/from_games/data/04_images.zip) | [Download 5](https://download.visinf.tu-darmstadt.de/data/from_games/data/05_images.zip) | [Download 6](https://download.visinf.tu-darmstadt.de/data/from_games/data/06_images.zip) | [Download 7](https://download.visinf.tu-darmstadt.de/data/from_games/data/07_images.zip) | [Download 8](https://download.visinf.tu-darmstadt.de/data/from_games/data/08_images.zip) | [Download 9](https://download.visinf.tu-darmstadt.de/data/from_games/data/09_images.zip) | [Download 10](https://download.visinf.tu-darmstadt.de/data/from_games/data/10_images.zip) | [Download 11](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip) | [Download 12](https://download.visinf.tu-darmstadt.de/data/from_games/data/02_labels.zip) | [Download 13](https://download.visinf.tu-darmstadt.de/data/from_games/data/03_labels.zip) | [Download 14](https://download.visinf.tu-darmstadt.de/data/from_games/data/04_labels.zip) | [Download 15](https://download.visinf.tu-darmstadt.de/data/from_games/data/05_labels.zip) | [Download 16](https://download.visinf.tu-darmstadt.de/data/from_games/data/06_labels.zip) | [Download 17](https://download.visinf.tu-darmstadt.de/data/from_games/data/07_labels.zip) | [Download 18](https://download.visinf.tu-darmstadt.de/data/from_games/data/08_labels.zip) | [Download 19](https://download.visinf.tu-darmstadt.de/data/from_games/data/09_labels.zip) | [Download 20](https://download.visinf.tu-darmstadt.de/data/from_games/data/10_labels.zip)

2. **Environment setup**:
    ```bash
    conda create -n benchmark-vfm-ss python=3.10
    conda activate benchmark-vfm-ss
    ```

3. **Install required packages**:
    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu123
    ```
    (replace with your CUDA version if not 12.3).

4. **Fine-tune a model**
   Here's an example for fine-tuning DINOv2 with the default setup on GPU 0 with 1 worker for data loading (replace ```/data``` with the folder where you stored the datasets)
   ```python main.py fit -c configs/ade20k_linear_semantic.yaml --root /data --data.num_workers 1 --trainer.devices [0] --model.network.encoder_name vit_base_patch14_dinov2```

## Reproducing results from the paper
    If using the BEiT models, download their checkpoints and convert them to timm format using ```convert_beit_ckpt.ipynb```.
    - **Base**: [Download](https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_224.pth)
    - **Large**: [Download](https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_224.pth)
    
    For the commands below, add `--root` to specify the path to where the datasets and checkpoints are stored and `--data.num_workers` to specify the number of workers for data     loading.
    
    Please note that compiling results in worse performance for EVA02 for some reason and BEiT models use sub norm.

    ### Default setup:  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

    ### Freezing the encoder:  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile --model.freeze_encoder True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile --model.freeze_encoder True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2 --model.freeze_encoder True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True --model.freeze_encoder True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli --model.freeze_encoder True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b --model.freeze_encoder True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k --model.freeze_encoder True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k --model.freeze_encoder True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae --model.freeze_encoder True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b --model.freeze_encoder True```  

    ### Changing the decoder:  
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```   
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
    ```python main.py fit -c configs/ade20k_mask2former_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

    ### Scaling the model:  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_large_patch14_clip_336.merged2b --no_compile```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_large_patch14_224.mim_m38m --no_compile```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch14_dinov2```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch16_224 --model.network.ckpt_path beit3_large_patch16_224.pth.timm --model.network.sub_norm True```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch16_siglip_384.webli```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch14_clip_224.dfn2b```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_large_patch16_384.fb_in22k_ft_in1k```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_large_patch16_384.fb_in1k```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_large_patch16_224.mae```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name samvit_large_patch16.sa1b```  

    ### Varying the patch size:  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile --model.network.patch_size 8```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2  --model.network.patch_size 8```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True --model.network.patch_size 8```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli --model.network.patch_size 8```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b --model.network.patch_size 8```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k --model.network.patch_size 8```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k --model.network.patch_size 8```   
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae --model.network.patch_size 8```  
    ```python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b --model.network.patch_size 8```  

    ### Changing the downstream dataset to PASCAL VOC:  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
    ```python main.py fit -c configs/pascal_voc_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

    ### Changing the downstream dataset to Cityscapes:  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
    ```python main.py fit -c configs/cityscapes_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

    ### Introducing a domain shift:  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name eva02_base_patch16_clip_224.merged2b --no_compile```  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name eva02_base_patch14_224.mim_in22k --no_compile```  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2```  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224 --model.network.ckpt_path beit3_base_patch16_224.pth.timm --model.network.sub_norm True```  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_siglip_512.webli```  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_clip_224.dfn2b```  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in22k_ft_in1k```  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name deit3_base_patch16_384.fb_in1k```  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name vit_base_patch16_224.mae```  
    ```python main.py fit -c configs/gta5_linear_semantic.yaml --model.network.encoder_name samvit_base_patch16.sa1b```  

## Acknowledgement
We borrow some code from Hugging Face Transformers (https://github.com/huggingface/transformers) (Apache-2.0 License)
