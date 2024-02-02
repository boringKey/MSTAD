import logging

from UniAD_Gradient.datasets.cifar_dataset import build_cifar10_dataloader
from UniAD_Gradient.datasets.custom_dataset import build_custom_dataloader, build_custom_dataset

logger = logging.getLogger("global")


def build(cfg, training, distributed):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed)
    elif dataset == "cifar10":
        data_loader = build_cifar10_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader

def build_dataset(cfg, training, distributed):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        # data_loader = build_custom_dataset(cfg, training, distributed)
        data_loader = build_custom_dataloader(cfg, training, distributed)
    elif dataset == "cifar10":
        data_loader = build_cifar10_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=True):
    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, training=True, distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, training=False, distributed=distributed)

    logger.info("build dataset done")
    return train_loader, test_loader

def build_datasets(cfg_dataset, distributed=True):
    train_dataset = None
    if cfg_dataset.get("train", None):
        train_dataset = build_dataset(cfg_dataset, training=True, distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_dataset = build_dataset(cfg_dataset, training=False, distributed=distributed)

    logger.info("build dataset done")
    return train_dataset, test_dataset
