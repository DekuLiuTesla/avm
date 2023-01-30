import itertools
from PIL import Image
import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
import argparse
import torch.multiprocessing as mp

"""
# RGB
COLOR = {
    "ego_vehicle": (0, 0, 0),
    "marker": (255, 255, 255),
    "curb": (0, 0, 255),
    "vehicle": (255, 0, 0),
    "unknown1": (0, 255, 0),
    "unknown2": (255, 255, 0),
}
"""

def map_mask(process_id, args):

    # BGR
    C = {
        (0, 0, 0) : 0,
        (255, 255, 255) : 1,
        (0, 0, 255) : 2,
        (255, 0, 0) : 3,
        (0, 255, 0) : 255,
        (0, 255, 255) : 255,
    }

    root = os.path.join(args.root, "gt")
    dest = os.path.join(args.root, "semantic")

    os.makedirs(dest, exist_ok=True)
    
    # RGB
    palette = itertools.chain(
        itertools.chain.from_iterable([(0, 0, 0), (255, 255, 255), (0, 0, 255)[::-1], (255, 0, 0)[::-1]]),
        itertools.chain.from_iterable(itertools.repeat((0, 255, 0), 256 - 4))
    )

    palette = list(palette)

    annos = os.listdir(root)

    length = len(annos)
    indices = range(process_id, length, args.nproc)

    for idx in tqdm(indices):
        anno = annos[idx]

        d = f"{dest}/{anno[:-4]}.png"
        if os.path.exists(d):
            continue

        mask = cv.imread(os.path.join(root, anno))
        new_mask = np.zeros(shape=mask.shape[:2], dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                new_mask[i, j] = C[tuple(mask[i, j])]
        
        new_mask = Image.fromarray(new_mask)
        new_mask.putpalette(palette)
        new_mask.save(d)

    print("OK")

def convert_split(args):

    # /images/00000221.jpg /gt/00000221.png

    def split(txt, mode):

        lines = open(f"{args.root}/{txt}", "r").read().splitlines()
        writer = open(f"{args.root}/{mode}.txt", "w")

        for line in lines:
            name = line.split(' ')[0].rstrip(".jpg").split('/')[-1]
            writer.write(f"{name}\n")
        
        print(f"{mode} created.")

    split(f"train_db.txt", "train")
    split(f"test_db.txt", "test")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="avm")
    parser.add_argument("--nproc", type=int, default=10)

    args = parser.parse_args()

    # Convert RGB to index
    mp.spawn(
        fn=map_mask, 
        args=(args, ),
        nprocs=args.nproc,
        join=True
    )

    # Convert split to voc-form
    convert_split(args)


    


