# ViT_LNAblation

Experimental study conducting various ablations to the LayerNorm present in the ViT architecture.

## Installation
To reproduce our results, run the following commands to set up the Python environment:
```
conda create -n DyT python=3.12
conda activate DyT
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install timm==1.0.15 tensorboard
```

## Training

To reproduce our results on ImageNet-100 with ViT-S , run the following commands: \

<details>
<summary>
Dynamic Tanh
</summary>
    
```
sudo python main.py --data_set IMNET --data_path /teamspace/studios/this_studio/dataset/imagenet-100 --enable_wandb true --project v
it_dys_adamw --dynamic_tanh true --batch_size 128 --model vit_small_patch16_224
```
</details>

<details>
<summary>
Dynamic Sigmoid
</summary>
    
```
sudo python main.py --data_set IMNET --data_path /teamspace/studios/this_studio/dataset/imagenet-100 --enable_wandb true --project v
it_dys_adamw --dynamic_sigmoid true --batch_size 128 --model vit_small_patch16_224
```
</details>
<details>
<summary>
Dynamic Softsign
</summary>
    
```
sudo python main.py --data_set IMNET --data_path /teamspace/studios/this_studio/dataset/imagenet-100 --enable_wandb true --project v
it_dys_adamw --dynamic_softsign true --batch_size 128 --model vit_small_patch16_224
```
</details>
<details>
<summary>
RMS Norm
</summary>
    
```
sudo python main.py --data_set IMNET --data_path /teamspace/studios/this_studio/dataset/imagenet-100 --enable_wandb true --project v
it_dys_adamw --rms_norm true --batch_size 128 --model vit_small_patch16_224
```
</details>

<details>
<summary>
Batch Norm
</summary>
    
```
sudo python main.py --data_set IMNET --data_path /teamspace/studios/this_studio/dataset/imagenet-100 --enable_wandb true --project v
it_dys_adamw --dynamic_softsign true --batch_size 128 --model vit_small_patch16_224
```
</details>

<details>
<summary>
Dynamic Tanh + AdamW
</summary>
    
```
sudo python main.py --data_set IMNET --data_path /teamspace/studios/this_studio/dataset/imagenet-100 --enable_wandb true --project v
it_dys_adamw --dynamic_tanh true --batch_size 128 --model vit_small_patch16_224
```
</details>

<details>
<summary>
Momentum
</summary>
    
```
sudo python main.py --data_set IMNET --data_path /teamspace/studios/this_studio/dataset/imagenet-100 --enable_wandb true --project v
it_dys_adamw --dynamic_tanh true --batch_size 128 --model vit_small_patch16_224 --opt momentum
```
</details>

<details>
<summary>
RMSProp
</summary>
    
```
sudo python main.py --data_set IMNET --data_path /teamspace/studios/this_studio/dataset/imagenet-100 --enable_wandb true --project v
it_dys_adamw --dynamic_tanh true --batch_size 128 --model vit_small_patch16_224 --opt rmsprop
```
</details>
