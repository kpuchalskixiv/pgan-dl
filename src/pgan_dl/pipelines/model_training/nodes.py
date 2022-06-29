from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch import nn
import torchvision as tv
import torch.utils.data as data_utils
from pytorch_lightning.loggers import WandbLogger
import wandb
from ...src.model import PGAN
from ...src.my_pgan import WGANGP_loss
import pytorch_lightning as pl


def create_data_loader(input_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    """
    Creates pytorch data loader from places365 dataset
    :param input_dir: Path to the root of the dataset
    :type input_dir: str
    :param batch_size: Number of images in a batch
    :type batch_size: int
    :param num_workers: How many subprocesses are used for data loading
    :type num_workers: int
    :return: Data loader for the dataset
    :rtype: torch.utils.data.DataLoader
    """

    data_train = DataLoader(
        tv.datasets.CIFAR10(input_dir, transform=transforms.ToTensor()),
        batch_size=batch_size, num_workers=num_workers
    )

    oneclass_data = []
    for d in data_train:
        data, labels = d
        for l in range(len(labels)):
            if labels[l] == 0:
                oneclass_data.append(data[l].view(1, 3, 32, 32))

    return DataLoader(
        data_utils.TensorDataset(torch.cat(oneclass_data, dim=0), torch.zeros(len(oneclass_data))),
        batch_size=batch_size, num_workers=num_workers
    )


def initialize(
    input_dir: str,
    latent_size: int,
    final_res: int,
    negative_slope: float,
    alpha_step: float,
    batch_size: int,
    lr: float,
    num_workers: int
) -> Tuple[PGAN, DataLoader]:
    """
    Initializes neural network for solving classification problem on Places365 dataset
    :param input_dir: Path to the split dataset.
    :type input_dir: str
    :param latent_size: A latent size of a model.
    :type latent_size: int
    :param final_res: A resolution of the generated image.
    :type final_res: int
    :param negative_slope: A hyperparam of the LeakyReLU activation function.
    :type negative_slope: float
    :param alpha_step: A alpha_step of the model
    :type alpha_step: float
    :param batch_size: Number of images in a batch.
    :type batch_size: int
    :param lr: Learning rate for model's optimizer.
    :type lr: float
    :param num_workers: How many subprocesses are used for data loading.
    :type num_workers: int
    :return: Model object alongside train, validation, test data loaders
    :rtype: Tuple[PGAN, torch.utils.data.DataLoader]
    """
    model = PGAN(
        lr=lr,
        latent_size=latent_size,
        final_res=final_res,
        activation_f=nn.LeakyReLU(negative_slope=negative_slope),
        alpha_step=alpha_step,
        loss_f=WGANGP_loss
    )

    dataloader = create_data_loader(input_dir=input_dir, batch_size=batch_size, num_workers=num_workers)

    return model, dataloader


def train_model(
    model: PGAN,
    dataloader: DataLoader,
    max_epochs: int,
    checkpoint_path: str,
    loger_entity: str,
    loger_name: str,
) -> PGAN:
    """Trains the model
    :param model: Model object created with initialize()
    :type model: PlacesModel
    :param dataloader: Pytorch dataloader which handles training data
    :type dataloader: torch.utils.data.DataLoader
    :param max_epochs: Max number of epochs
    :type max_epochs: number
    :param checkpoint_path: Path to directory in which to save model checkpoints
    :type checkpoint_path: str
    :param loger_entity: WandDB entity name
    :type loger_entity: str
    :param loger_name: WandDB loger name
    :type loger_name: str

    :return: A trained model
    :rtype: PGAN
    """
    wandb.init()
    wandb_logger = WandbLogger(project="PGAN",  name=loger_name, entity=loger_entity)
    gpu_devices = 1 if torch.cuda.is_available() else 0

    trainer = pl.Trainer(
        gpus=gpu_devices,
        precision=16,
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path)]
    )
    trainer.fit(model, dataloader)

    wandb.finish()

    return model, dataloader
