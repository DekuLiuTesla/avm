from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
import argparse

def get_model(args):

    PALETTE = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]]
    model = init_segmentor(args.config, args.checkpoint, device='cuda')
    model.PALETTE = PALETTE

    return model

def inference_sample(args):

    model = get_model(args)

    img = mmcv.imread(args.image_path)
    result = inference_segmentor(model, img)
    name = args.image_path.split('/')[-1].split('.')[0] + ".png"
    out_file = os.path.join(args.save_dir, name)

    if args.rgb:
        model.show_result(img, result, out_file=out_file, opacity=1)
    else:
        mmcv.imwrite(result[0], out_file)




if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="upernet_swin_small_patch4_window7_512x512_160k_avm20k_pretrain_224x224_1K.py")
    parser.add_argument("--checkpoint", type=str, default="/home/zeyu_yan/openmmlab/work_dirs/swin_avm_demo/iter_128000.pth")
    parser.add_argument("--image_path", type=str, default="demo.jpg")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--rgb", type=lambda x: x == "True", default="True", choices=["True", "False"])

    args = parser.parse_args()

    inference_sample(args)

