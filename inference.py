from mmseg.apis import init_segmentor
import mmcv
import os
import argparse
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

def get_model(args):

    PALETTE = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]]
    model = init_segmentor(args.config, args.checkpoint, device='cuda')
    model.PALETTE = PALETTE

    return model

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def inference_segmentor(model, imgs):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]

    # change img_scale
    if args.shape is not None:
        if cfg.data.test.pipeline[1]["type"] == "MultiScaleFlipAug":
            cfg.data.test.pipeline[1]["img_scale"] = (args.shape[1], args.shape[0])
        else:
            raise TypeError("Only support MultiScaleFlipAug")

    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    for img in imgs:
        img_data = dict(img=img)
        img_data = test_pipeline(img_data)
        data.append(img_data)
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def inference_sample(args):

    model = get_model(args)

    img = mmcv.imread(args.image_path)
    result = inference_segmentor(model, img)
    name = args.image_path.split('/')[-1].split('.')[0] + ".png"
    out_file = os.path.join(args.save_dir, name)

    if args.rgb:
        model.show_result(img, result, out_file=out_file, opacity=0.5)
    else:
        mmcv.imwrite(result[0], out_file)


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/bisenetv2_fcn_4x4_1024x1024_160k_avm.py")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/iter_144000.pth")
    parser.add_argument("--image_path", type=str, default="demo.jpg")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--rgb", type=lambda x: x == "True", default="True", choices=["True", "False"])
    parser.add_argument("--shape", type=int, default=None, nargs='+', help="img_scale: [H x W]")

    args = parser.parse_args()

    inference_sample(args)

