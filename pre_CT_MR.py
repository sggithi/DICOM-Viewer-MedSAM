# -*- coding: utf-8 -*-
# %% import packages
import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
import cc3d
import multiprocessing as mp
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-modality", type=str, default="CT", help="CT or MR, [default: CT]")
parser.add_argument("-anatomy", type=str, default="Abd", help="Anatomy name, [default: Abd]")
parser.add_argument("-gt_name_suffix", type=str, default=".nii.gz", help="Suffix of the ground truth name, [default: .nii.gz]")
parser.add_argument("-img_path", type=str, default="data/images", help="Path to the nii images, [default: data/images]")
parser.add_argument("-gt_path", type=str, default="data/labels", help="Path to the ground truth, [default: data/labels]")
parser.add_argument("-output_path", type=str, default="data/npz", help="Path to save the npy files, [default: ./data/npz]")
parser.add_argument("-num_workers", type=int, default=4, help="Number of workers, [default: 4]")
parser.add_argument("-window_level", type=int, default=50, help="CT window level, [default: 40]")
parser.add_argument("-window_width", type=int, default=250, help="CT window width, [default: 400]")
parser.add_argument("--save_nii", action="store_true", help="Save the image and ground truth as nii files for sanity check; they can be removed")

args = parser.parse_args()

modality = args.modality
anatomy = args.anatomy
gt_name_suffix = args.gt_name_suffix
prefix = modality + "_" + anatomy + "_"

nii_path = args.img_path
gt_path = args.gt_path
output_path = args.output_path
npz_tr_path = os.path.join(output_path, "MedSAM_train", prefix[:-1])
os.makedirs(npz_tr_path, exist_ok=True)
npz_ts_path = os.path.join(output_path, "MedSAM_test", prefix[:-1])
os.makedirs(npz_ts_path, exist_ok=True)

num_workers = args.num_workers
voxel_num_thre2d = 100
voxel_num_thre3d = 1000

names = sorted(os.listdir(gt_path))
print(f"Original # files {len(names)}")
print(f"Original names: {names}")

# Add debug output to check the existence of corresponding image files
names = [name for name in names if os.path.exists(os.path.join(nii_path, name.split(gt_name_suffix)[0] + ".nii.gz"))]
print(f"After sanity check # files {len(names)}")
print(f"After sanity check names: {names}")

remove_label_ids = [12]
tumor_id = None
WINDOW_LEVEL = args.window_level
WINDOW_WIDTH = args.window_width
save_nii = args.save_nii

def preprocess(name, npz_path):
    image_name = name.split(gt_name_suffix)[0] + ".nii.gz"
    gt_name = name.split(".nii.gz")[0] + gt_name_suffix
    gt_sitk = sitk.ReadImage(os.path.join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        tumor_inst, tumor_n = cc3d.connected_components(tumor_bw, connectivity=26, return_N=True)
        gt_data_ori[tumor_inst > 0] = tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1

    gt_data_ori = cc3d.dust(gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True)
    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        gt_data_ori[slice_i, :, :] = cc3d.dust(gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True)

    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) > 0:
        gt_roi = gt_data_ori[z_index, :, :]
        img_sitk = sitk.ReadImage(os.path.join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        if modality == "CT":
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
        else:
            lower_bound, upper_bound = np.percentile(image_data[image_data > 0], 0.5), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
            image_data_pre[image_data == 0] = 0

        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]
        np.savez_compressed(os.path.join(npz_path, prefix + gt_name.split(gt_name_suffix)[0] + '.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())

        if save_nii:
            img_roi_sitk = sitk.GetImageFromArray(img_roi)
            img_roi_sitk.SetSpacing(img_sitk.GetSpacing())
            sitk.WriteImage(img_roi_sitk, os.path.join(npz_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"))
            gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
            gt_roi_sitk.SetSpacing(img_sitk.GetSpacing())
            sitk.WriteImage(gt_roi_sitk, os.path.join(npz_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"))

# Split data into training and testing sets with an 8:2 ratio
split_index = int(0.8 * len(names))
tr_names = names[:split_index]
ts_names = names[split_index:]

print(f"Training set size: {len(tr_names)}")
print(f"Testing set size: {len(ts_names)}")

if __name__ == "__main__":
    preprocess_tr = partial(preprocess, npz_path=npz_tr_path)
    preprocess_ts = partial(preprocess, npz_path=npz_ts_path)

    with mp.Pool(num_workers) as p:
        with tqdm(total=len(tr_names)) as pbar:
            pbar.set_description("Preprocessing training data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, tr_names))):
                pbar.update()
        with tqdm(total=len(ts_names)) as pbar:
            pbar.set_description("Preprocessing testing data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_ts, ts_names))):
                pbar.update()
