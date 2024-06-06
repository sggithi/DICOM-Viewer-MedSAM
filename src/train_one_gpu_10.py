import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Set OpenCV backend to non-GUI

import random
import monai
from os import makedirs
from os.path import join, exists, isfile, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_root", type=str, default=None,
    help="Path to the npy data root."
)
parser.add_argument(
    "-pretrained_checkpoint", type=str, default="work_dir/LiteMedSAM/lite_medsam.pth",
    help="Path to the pretrained Lite-MedSAM checkpoint."
)
parser.add_argument(
    "-resume", type=str, default='workdir/medsam_lite_latest.pth',
    help="Path to the checkpoint to continue training."
)
parser.add_argument(
    "-work_dir", type=str, default="./workdir/bbox_10",
    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument(
    "-num_epochs", type=int, default=10,
    help="Number of epochs to train."
)
parser.add_argument(
    "-batch_size", type=int, default=4,
    help="Batch size."
)
parser.add_argument(
    "-num_workers", type=int, default=8,
    help="Number of workers for dataloader."
)
parser.add_argument(
    "-device", type=str, default="cuda:0",
    help="Device to train on."
)
parser.add_argument(
    "-bbox_shift", type=int, default=5,
    help="Perturbation to bounding box coordinates during training."
)
parser.add_argument(
    "-lr", type=float, default=0.00005,
    help="Learning rate."
)
parser.add_argument(
    "-weight_decay", type=float, default=0.01,
    help="Weight decay."
)
parser.add_argument(
    "-iou_loss_weight", type=float, default=1.0,
    help="Weight of IoU loss."
)
parser.add_argument(
    "-seg_loss_weight", type=float, default=1.0,
    help="Weight of segmentation loss."
)
parser.add_argument(
    "-ce_loss_weight", type=float, default=1.0,
    help="Weight of cross entropy loss."
)
parser.add_argument(
    "--sanity_check", action="store_true",
    help="Whether to do sanity check for dataloading."
)

args = parser.parse_args()
data_root = "../data/npy"
test_data_root = "../data/test"
# Adjust data_root to be relative to the script location if not provided
if args.data_root is None:
    data_root = join(os.path.dirname(__file__), '..', 'data', 'npy')
else:
    data_root = args.data_root

# Ensure data_root is not None and check directories
if data_root is None:
    raise ValueError("data_root cannot be None")

# Check if the directories exist
gts_path = join(test_data_root, 'gts')
imgs_path = join(test_data_root, 'imgs')


if not exists(gts_path):
    raise ValueError(f"GT directory does not exist: {gts_path}")
if not exists(imgs_path):
    raise ValueError(f"IMG directory does not exist: {imgs_path}")

# List files in the directories
gt_files = sorted(glob(join(gts_path, '*.npy')))
img_files = sorted(glob(join(imgs_path, '*.npy')))

# print(f"GT files: {gt_files}")
# print(f"IMG files: {img_files}")

# Check if there are no files
if len(gt_files) == 0 or len(img_files) == 0:
    raise ValueError(f"No .npy files found in the specified directories: {gts_path} or {imgs_path}")

# Ensure that image and GT filenames match
for gt_file, img_file in zip(gt_files, img_files):
    if basename(gt_file) != basename(img_file):
        raise ValueError(f"Filename mismatch between GT and IMG files: {gt_file} and {img_file}")

# Define the NpyDataset class
class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=256, bbox_shift=5, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_files = sorted(glob(join(self.gt_path, '*.npy')))
        self.img_files = sorted(glob(join(self.img_path, '*.npy')))
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
    
    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        gt_file = self.gt_files[index]
        img_name = basename(img_file)
        gt_name = basename(gt_file)

        assert img_name == gt_name, f'Image and GT name mismatch: {img_name} vs {gt_name}'

        img_3c = np.load(img_file, 'r', allow_pickle=True)
        img_resize = self.resize_longest_side(img_3c)
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None)
        img_padded = self.pad_image(img_resize)
        img_padded = np.transpose(img_padded, (2, 0, 1))
        assert np.max(img_padded) <= 1.0 and np.min(img_padded) >= 0.0, 'Image should be normalized to [0, 1]'

        gt = np.load(gt_file, 'r', allow_pickle=True)
        gt = cv2.resize(gt, (img_resize.shape[1], img_resize.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        gt = self.pad_image(gt)
        label_ids = np.unique(gt)[1:]
        
        try:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt))

        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))

        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = gt2D.shape
        # x_min = max(0, x_min - W*0.05)
        # x_max = min(W-1, x_max +  W*0.05)
        # y_min = max(0, y_min - H*0.05)
        # y_max = min(H-1, y_max + H*0.05)      

        x_min = max(0, x_min - W*0.1)
        x_max = min(W-1, x_max +  W*0.1)
        y_min = max(0, y_min - H*0.1)
        y_max = min(H-1, y_max + H*0.1)     
        # x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        # x_max = min(W-1, x_max + random.randint(0, self.bbox_shift))
        # y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        # y_max = min(H-1, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max]) # bound box shape
        #bboxes = np.array([0, 0, 256, 256])


        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(),
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

    def resize_longest_side(self, image):
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3:
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else:
            image_padded = np.pad(image, ((0, padh), (0, padw)))
        return image_padded

# Continue with your existing dataset initialization and training code
train_dataset = NpyDataset(data_root=data_root, data_aug=True)
test_dataset = NpyDataset(data_root=test_data_root, data_aug=True)
# Check for empty dataset
if len(train_dataset) == 0:
    raise ValueError(f"No .npy files found in the specified directory: {data_root}")
if len(test_dataset) == 0:
    raise ValueError(f"No .npy files found in the specified directory: {test_data_root}")
print("test", len(test_dataset))    
# Prepare directories and parameters
work_dir = args.work_dir
print("directory", work_dir)
medsam_lite_checkpoint = args.pretrained_checkpoint
print("checkpoint", medsam_lite_checkpoint)
num_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.num_workers
device = args.device
bbox_shift = args.bbox_shift
lr = args.lr
weight_decay = args.weight_decay
iou_loss_weight = args.iou_loss_weight
seg_loss_weight = args.seg_loss_weight
ce_loss_weight = args.ce_loss_weight
do_sancheck = args.sanity_check
checkpoint = args.resume

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
makedirs(work_dir, exist_ok=True)

torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def cal_iou(result, reference):
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    iou = intersection.float() / union.float()
    return iou.unsqueeze(1)

# Sanity check for dataset class
if do_sancheck:
    tr_dataset = NpyDataset(data_root, data_aug=True)
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
    for step, batch in enumerate(tr_dataloader):
        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        idx = random.randint(0, 4)
        image = batch["image"]
        gt = batch["gt2D"]
        bboxes = batch["bboxes"]
        names_temp = batch["image_name"]
        axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
        show_box(bboxes[idx].numpy().squeeze(), axs[0])
        axs[0].axis('off')
        axs[0].set_title(names_temp[idx])
        idx = random.randint(4, 7)
        axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
        show_box(bboxes[idx].numpy().squeeze(), axs[1])
        axs[1].axis('off')
        axs[1].set_title(names_temp[idx])
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            join(work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        break

class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        masks = masks[:, :, :new_size[0], :new_size[1]]
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks

medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[64, 128, 160, 320],
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
    image_encoder=medsam_lite_image_encoder,
    mask_decoder=medsam_lite_mask_decoder,
    prompt_encoder=medsam_lite_prompt_encoder
)

if medsam_lite_checkpoint is not None:
    if isfile(medsam_lite_checkpoint):
        print(f"Finetuning with pretrained weights {medsam_lite_checkpoint}")
        medsam_lite_ckpt = torch.load(medsam_lite_checkpoint, map_location="cpu")
        #print("epoch", medsam_lite_checkpoint['epoch'])
        medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
        ######### 
        #medsam_lite_model.load_state_dict(medsam_lite_ckpt["model"], strict=True)
    else:
        print(f"Pretained weights {medsam_lite_checkpoint} not found, training from scratch")
        

medsam_lite_model = medsam_lite_model.to(device)
medsam_lite_model.train()

print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")

optimizer = optim.AdamW(
    medsam_lite_model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.9,
    patience=5,
    cooldown=0
)
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
iou_loss = nn.MSELoss(reduction='mean')

if checkpoint and isfile(checkpoint):
    print(f"Resuming from checkpoint {checkpoint}")
    checkpoint = torch.load(checkpoint)
    medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["loss"]
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    start_epoch = 0
    best_loss = 1e10

train_losses = []
test_losses =[]
for epoch in range(start_epoch + 1, num_epochs + 1):
    epoch_loss = [1e10 for _ in range(len(train_loader))]
    epoch_iou_loss =  [1e10 for _ in range(len(train_loader))]
    epoch_test_iou_loss = [1e10 for _ in range(len(test_loader))]
    epoch_start_time = time()
    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        image = batch["image"]
        gt2D = batch["gt2D"]
        boxes = batch["bboxes"]
        optimizer.zero_grad()
        image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
        logits_pred, iou_pred = medsam_lite_model(image, boxes)
        l_seg = seg_loss(logits_pred, gt2D)
        l_ce = ce_loss(logits_pred, gt2D.float())
        mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
        iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
        l_iou = iou_loss(iou_pred, iou_gt)
        ## loss function######################################
        loss = mask_loss + iou_loss_weight * l_iou 
        # loss = l_iou
        ########################################################
        epoch_loss[step] = loss.item()
        epoch_iou_loss[step] = l_iou.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}, iou_loss:{l_iou}")
    
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
            logits_pred, iou_pred = medsam_lite_model(image, boxes)
            iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
            iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            loss = mask_loss + iou_loss_weight * l_iou 
            epoch_test_iou_loss[step] = loss.item()

    epoch_test_iou_reduced = sum(epoch_test_iou_loss) / len(epoch_test_iou_loss)   
    
    epoch_end_time = time()
    epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
    epoch_iou_reduced = sum(epoch_iou_loss) / len(epoch_iou_loss)
    print(f"Epoch{epoch}, loss {epoch_loss_reduced}, iou_loss{epoch_iou_reduced} test loss {epoch_test_iou_reduced}")
    test_losses.append(epoch_test_iou_reduced)
    train_losses.append(epoch_loss_reduced)
    lr_scheduler.step(epoch_loss_reduced)
    model_weights = medsam_lite_model.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": best_loss,
    }
    ## test loss ###########
     
    ###################################################################### 
    # basic : minimum / 5 % / image size
    torch.save(checkpoint, join(work_dir, "medsam_lite_latest.pth"))
    if epoch_loss_reduced < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}, iou loss:{epoch_iou_reduced:.4f}")
        best_loss = epoch_loss_reduced
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "medsam_lite_best.pth"))
    ########################################################################
    epoch_loss_reduced = 1e10
    # Plot loss
    plt.plot(train_losses, label = "Train")
    plt.plot(test_losses, label='Test')
    plt.title("Dice + Binary Cross Entropy + IoU Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(work_dir, "train_loss.png"))
    plt.close()
