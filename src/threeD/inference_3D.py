from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from matplotlib import pyplot as plt
import cv2
import torch.multiprocessing as mp

import argparse


#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2023) #Sets the seed for generating random numbers on the CPU. 
torch.cuda.manual_seed(2023)  #Set the seed for generating random numbers for the current GPU
np.random.seed(2023)

# medsam_lite_checkpoint_path = "./work_dir/LiteMedSAM/lite_medsam.pth"
###################################
#medsam_lite_checkpoint_path = "./work_dir/LiteMedSAM/medsam_lite_best_image_size.pth"#
#medsam_lite_checkpoint_path = "./work_dir/LiteMedSAM/medsam_lite_best_basic.pth"
#medsam_lite_checkpoint_path = "./work_dir/LiteMedSAM/medsam_lite_best_5.pth"
medsam_lite_checkpoint_path = "./work_dir/LiteMedSAM/medsam_lite_best_10.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bbox_shift = 5
image_size = 256


def resize_longest_side(image, target_length):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder



@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, height, width):
    """
    Perform segmentation using MedSAM

    Parameters
    ----------
    img_embed : np.ndarray
        The image to segment, a tensor of shape (B, 256, 64, 64) 
        where B is the batch size. 256 is the number of channels and 64 x 64 is the spatial resolution.

    box_256 : np.ndarray
        The bounding box coordinates at 256 scale, typically a tensor of shape (B, 4). 
        The bounding box coordinates are in the format (x_min, y_min, x_max, y_max).
    height : int
        The height of the original image
    width : int
        The width of the original image

    Returns
    -------
    np.ndarray
        The segmentation mask
    """

    # Convert Bounding box to tensor
    box_torch = torch.as_tensor(box_256, dtype=torch.float, device=img_embed.device) # box_256 is converted to a tensor box_torch on the same device as img_embed.
    # Adjust the shape of box_torch
    if len(box_torch.shape) == 2:  # If box_torch has shape (B, 4), it is reshaped to (B, 1, 4) by adding an extra dimension.
        box_torch = box_torch[:, None, :]  # (B, 1, 4)  

    # Encode prompts: bounding box only
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder( 
        # prompt_encoder encodes the bounding box box_torch into sparse_embeddings and dense_embeddings.
        # sparse_embeddings shape: (B, 2, 256)
	    # dense_embeddings shape: (B, 256, 64, 64)
        points=None,
        boxes=box_torch,
        masks=None,
    )

    # Decode masks using the image embeddings and prompt embeddings
    low_res_logits, _ = medsam_model.mask_decoder( # low_res_logits shape: (B, 1, 256, 256)
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    # Sigmoid activation: convert logits to probabilities
    low_res_pred = torch.sigmoid(low_res_logits)  # (B, 1, 256, 256)

    #Interpolate the low resolution prediction to the original image size
    low_res_pred = F.interpolate( # low_res_pred shape: (B, 1, height, width)
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  

    ######### Output size (256, 256 ###############################)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (height, width) = (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8) # binary mask medsam_seg is created where values greater than 0.5 are set to 1 (foreground), and the rest are set to 0 (background).

    return medsam_seg


medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
)

medsam_lite_model = MedSAM_Lite(
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

medsam_lite_checkpoint = torch.load(medsam_lite_checkpoint_path, map_location='cpu')
############################################## Fine Tuning Model
medsam_lite_model.load_state_dict(medsam_lite_checkpoint["model"], strict=True)
################################################################
# medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
medsam_lite_model.to(device)
medsam_lite_model.eval()
