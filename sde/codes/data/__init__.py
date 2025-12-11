"""create dataset and dataloader"""

import logging

import torch
import torch.utils.data


def create_dataloader(dataset, opt, sampler=None):
    if dataset.train:
        num_workers = opt["datasets"]["n_workers"]
        batch_size = opt["datasets"]["batch_size"]
        shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
        )
    else:
        batch_size = opt["datasets"]["batch_size"]
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )


def create_dataset(dataset_opt, train=True):
    dataset_mode = dataset_opt["dataset"]
    if dataset_mode == "chex":  # Predictor
        from data.chex_dataset import ChexDataset as D

        dataset = D(dataset_opt, train=train)
    elif dataset_mode == "ucsf":  # SFTMD
        from data.ucsf_dataset import UcsfDataset as D

        dataset = D(dataset_opt, train=train)
    else:
        raise NotImplementedError(
            "Dataset [{:s}] is not recognized.".format(dataset_mode)
        )

    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_mode
        )
    )
    return dataset


def create_test_dataset(dataset_opt):
    dataset_mode = dataset_opt["dataset"]
    if dataset_mode == "chex":  # Predictor
        from data.chex_dataset import ChexDataset as D

        dataset = D(dataset_opt, train=False, test=True)
    return dataset
