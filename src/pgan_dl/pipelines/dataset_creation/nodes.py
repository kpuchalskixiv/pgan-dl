import torchvision as tv
from torchvision import transforms


def get_data(out_data_dir: str):
    """Downloads dataset from torch vision datasets and saves it locally
    :param out_data_dir: Path to the root of processed dataset (where to save data)
    :type out_data_dir: str
    """
    tv.datasets.CIFAR10(out_data_dir, transform=transforms.ToTensor(), download=True)
