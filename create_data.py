# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import glob
import warnings
import matplotlib.pyplot as plt

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmseg.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='create cityscape-like dataset')
    parser.add_argument(
        '--avm-dir',
        default=False,
        type=str,
        help='Directory of AVM dataset')
    parser.add_argument(
        '--output-dir',
        default='./output',
        type=str,
        help='Directory of output dataset')
    args = parser.parse_args()
    return args


def save_img(avm_dir, output_dir):
    img_suffix = '_leftImg8bit.png'
    output_img_path = os.path.join(output_dir, 'leftImg8bit/train')
    avm_img_path = os.path.join(avm_dir, 'images')
    print('Saving Images ...')

    filename_list = os.listdir(avm_img_path)
    for img_filename in filename_list:
        avm_img_file = os.path.join(avm_img_path, img_filename, 'avm')
        img_files = sorted(os.listdir(avm_img_file))

        print(f'\nProcessing {img_filename}: ')
        progress_bar = mmcv.ProgressBar(len(img_files))
        for img_name in img_files:
            img_path = os.path.join(avm_img_file, img_name)
            img = mmcv.imread(img_path)

            save_filename = img_filename + '_' + img_name.split('.')[0] + img_suffix
            save_path = os.path.join(output_img_path, save_filename)
            mmcv.imwrite(img, save_path)
            progress_bar.update()

    print('\nDone')


def save_anno(avm_dir, output_dir):
    img_suffix = '_gtFine_labelTrainIds.png'
    seg_map_suffix = '_gtFine_labelTrainIds.png'
    output_img_path = os.path.join(output_dir, 'leftImg8bit/train')
    output_anno_path = os.path.join(output_dir, 'gtFine/train')
    avm_img_path = os.path.join(avm_dir, 'images')
    avm_anno_path = os.path.join(avm_dir, 'mask')
    print('Saving Annotations ...')

    filename_list = os.listdir(avm_img_path)
    for img_filename in filename_list:
        avm_img_file = os.path.join(avm_img_path, img_filename, 'avm')
        img_files = sorted(os.listdir(avm_img_file))

        avm_anno_file = os.path.join(avm_anno_path, img_filename)
        anno_files = sorted(os.listdir(avm_anno_file))

        print(f'\nProcessing {img_filename}: ')
        progress_bar = mmcv.ProgressBar(len(img_files))
        for img_name, anno_name in zip(img_files, anno_files):
            img_path = os.path.join(avm_img_file, img_name)
            img = mmcv.imread(img_path)

            anno_path = os.path.join(avm_anno_file, anno_name)
            anno_img = mmcv.imread(anno_path)[..., -1]

            save_filename = img_filename + '_' + img_name.split('.')[0] + seg_map_suffix
            save_path = os.path.join(output_anno_path, save_filename)
            mmcv.imwrite(anno_img, save_path)
            progress_bar.update()

    print('\nDone')

def vis_results(avm_dir, output_dir):
    save_dir = output_dir
    out_file = 'out_file_cityscapes'
    image = mmcv.imread(os.path.join(output_dir, 'leftImg8bit/train/b3_to_b2_0_leftImg8bit.png'))
    sem_seg = mmcv.imread(os.path.join(output_dir, 'gtFine/train/b3_to_b2_0_gtFine_labelTrainIds.png'))

    plt.imshow(sem_seg[..., -1], cmap='jet')
    plt.colorbar()
    plt.show()
    plt.imshow(image)
    plt.show()


def main():
    args = parse_args()
    avm_dir = args.avm_dir
    output_dir = args.output_dir

    save_img(avm_dir, output_dir)
    save_anno(avm_dir, output_dir)
    # vis_results(avm_dir, output_dir)


if __name__ == '__main__':
    main()
