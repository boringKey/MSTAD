import argparse
import glob
import importlib
import os
import pprint

import cv2
import numpy as np
import torch
import torch.optim
import yaml
from easydict import EasyDict
from utils.misc_helper import create_logger

parser = argparse.ArgumentParser(description="UniAD")
parser.add_argument("--config", default="/home/scu-its-gpu-001/UniAD_Gradient/experiments/vis_recon/config.yaml")
parser.add_argument("--class_name", default="zipper")


def update_config(config):
    # update planes & strides
    backbone_path, backbone_type = config.net[0].type.rsplit(".", 1)
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outplanes = []
    for layer in config.net[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        outplanes.append(backbone["planes"][idx])

    config.net[2].kwargs.instrides = config.net[1].kwargs.outstrides
    config.net[2].kwargs.inplanes = [sum(outplanes)]
    return config


def load_state_decoder(path, model):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)
        state_dict = checkpoint["state_dict"]

        # state_dict of decoder
        state_dict_decoder = {}
        for k, v in state_dict.items():
            if "reconstruction." in k:
                k_new = k.replace("reconstruction.", "")
                state_dict_decoder[k_new] = v

        # fix size mismatch error
        ignore_keys = []
        for k, v in state_dict_decoder.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    print(
                        "caution: size-mismatch key: {} size: {} -> {}".format(
                            k, v.shape, v_dst.shape
                        )
                    )

        for k in ignore_keys:
            state_dict_decoder.pop(k)

        model.load_state_dict(state_dict_decoder, strict=False)

        ckpt_keys = set(state_dict_decoder.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print("caution: missing keys from checkpoint {}: {}".format(path, k))
    else:
        print("=> no checkpoint found at '{}'".format(path))


def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    update_config(config)
    config.saver.load_path = config.saver.load_path.replace(
        "{class_name}", args.class_name
    )

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

    logger = create_logger(
        "global_logger", config.log_path + "/dec_{}.log".format(args.class_name)
    )
    logger.info("args: {}".format(pprint.pformat(args)))
    logger.info("config: {}".format(pprint.pformat(config)))

    # create model
    # config.net[2] 直接跳过前两个模型 创建ResNet模型
    module_name, cls_name = config.net[2].type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model = getattr(module, cls_name)(**config.net[2].kwargs)
    load_state_decoder(config.saver.load_path, model)
    model.cuda()

    mean = (
        torch.tensor(config.data.pixel_mean).cuda().unsqueeze(0).unsqueeze(0)
    )  # 1 x 1 x 3
    std = (
        torch.tensor(config.data.pixel_std).cuda().unsqueeze(0).unsqueeze(0)
    )  # 1 x 1 x 3

    # 对 从transformer结构提取出的指定{class_name}的 feature信息进行重建
    # feature_paths = {list}
    # 包含所有capsule的npy文件路径 如：'/home/scu-its-gpu-001/PycharmProjects/UniAD/experiments/MVTec-AD/result_recon/capsule/good/013.npy'
    feature_paths = glob.glob(
        os.path.join(config.data.feature_dir, args.class_name, "*/*.npy")
    )
    feature_paths = sorted(feature_paths)
    # for feature_path in feature_paths:
    for i in range(0, len(feature_paths), 4):
        feature_list = []
        feature_path = feature_paths[i:i+4]
        for feature in feature_path:
        # feature = (1, 272, 14, 14) 即为EfficentNet 提取出来的特征图
            feature = np.load(feature)
            feature = torch.tensor(feature).cuda().unsqueeze(0)
            input = {"feature_align": feature}
            with torch.no_grad():
                output = model(input)
            image_rec = (
                output["image_rec"].squeeze(0).permute(1, 2, 0)
            )  # 1 x 3 x h x w -> h x w x 3
            image_rec = (image_rec * std + mean) * 255
            image_rec = image_rec.cpu().numpy()
            feature_list.append(image_rec)
        # image
        filedir, filename = os.path.split(feature_path[0])
        filename = filename.split('_')[0]
        filedir, defename = os.path.split(filedir)
        _, clsname = os.path.split(filedir)
        filename_, _ = os.path.splitext(filename)
        # '/home/scu-its-gpu-001/UniAD_origin/data/MVTec-AD/mvtec_anomaly_detection/capsule/test/poke/004_DecoderLayer1.png'
        imagepath = os.path.join(
            config.data.dataset_dir, clsname, "test", defename, filename_ + ".png"
        )
        image = cv2.imread(imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (config.data.input_size[1], config.data.input_size[0])
        )  # h,w -> w,h

        image = np.concatenate([image, feature_list[0],feature_list[1],feature_list[2],feature_list[3]],
                               axis=1)  # 2h x w x 3
        vis_recon_dir = os.path.join(config.save_path, clsname, defename)
        os.makedirs(vis_recon_dir, exist_ok=True)
        savepath = os.path.join(vis_recon_dir, filename_ + ".jpg")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepath, image)

        print(f"Success: Feature: {feature_path}\n         Saved: {savepath}")


if __name__ == "__main__":
    main()
