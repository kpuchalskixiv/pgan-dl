from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
from ...src.model import PGAN
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from scipy import linalg

device = 'cuda' if torch.cuda.is_available() else 'cpu'


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LatentVectorDataset(Dataset):
    def __init__(self, samples_no=10000):
        self.samples_no = samples_no

    def __len__(self):
        return self.samples_no

    def __getitem__(self, idx):
        return F.normalize(torch.rand(512), p=2, dim=0).to(device)


def compute_embeddings(inception, dataloader, pgan=None):
    res = []
    for batch in tqdm(dataloader):
        images = pgan.generator(batch) if pgan else batch[0]
        preprocessed = [preprocess(image) for image in images]
        input = torch.stack(preprocessed).to(device)

        embeddings = inception(input)
        res.append(embeddings)

    res = torch.stack(res)
    return res.detach().cpu().numpy()


def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def prepare_inception_embedder():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

    model.fc = Identity()
    model.to(device)
    model.eval()
    return model


def evaluate(model: PGAN, dataloader: DataLoader, generated_samples_no: int, batch_size: int):
    """Computes a FID score on given dataset

    :param model: pre-trained PGAN model
    :type model: PGAN
    :param dataloader: dataloader used for training
    :type dataloader: torch.utils.data.DataLoader
    :param generated_samples_no: number of samples to be generated and used for evaluation
    :type generated_samples_no: int
    :param batch_size: Number of images in a batch
    :type batch_size: int
    """
    model.to(device)
    model.generator.to(device)
    model.generator.eval()
    inception = prepare_inception_embedder()

    #batch_size = 4

    # compute embeddings for real images
    real_image_embeddings = compute_embeddings(inception, dataloader)

    # compute embeddings for generated images
    gen_dataset = LatentVectorDataset(generated_samples_no)
    gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size)

    generated_image_embeddings = compute_embeddings(inception, gen_dataloader, pgan=model)

    fid = calculate_fid(real_image_embeddings, generated_image_embeddings)

    print()
    print('/'*100)
    print(f'FID: {fid}')
    print('/'*100)

    return fid
