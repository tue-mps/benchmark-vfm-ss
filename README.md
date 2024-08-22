## Getting started
1. **Download dataset.**
    - **Cityscapes**: [Download 1](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [Download 2](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
    
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
   Here's an example for fine-tuning DINOv2 on Cityscapes with the default setup on GPU 0 with 1 worker for data loading:
   ```bash
   python main.py fit -c configs/cityscapes_linear_semantic.yaml --root /data --data.num_workers 1 --trainer.devices [0] --model.network.encoder_name vit_base_patch14_dinov2
   ```
   (replace ```/data``` with the folder where you stored the datasets)  

## Reproducing results from the paper
For the commands below, add `--root` to specify the path to where the datasets and checkpoints are stored and `--data.num_workers` to specify the number of workers for data     loading.

## Acknowledgement
We borrow some code from Hugging Face Transformers (https://github.com/huggingface/transformers) (Apache-2.0 License)
