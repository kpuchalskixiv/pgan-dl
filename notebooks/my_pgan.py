import torch
from torch import nn
from pytorch_lightning import LightningModule
import torchvision as tv
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as data_utils

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from collections import OrderedDict

from custom_layers import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d
from gradient_losses import WGANGPGradientPenalty

from pytorch_lightning.loggers import WandbLogger
import wandb


class Gen_residual(nn.Module):
    def __init__(self, res, channels, device='cuda', start_alpha=0.1, activation_f=nn.LeakyReLU(), normalize=True):
        super().__init__()
        self.alpha=start_alpha
        self.channels=channels
        self.f=activation_f
        self.device=device

        if normalize:
            self.normalization_layer=NormalizationLayer
        else:
            self.normalization_layer=nn.Identity

        if res>=64:
            self.model=nn.ModuleList([
                        nn.Upsample(scale_factor=(2,2), mode='nearest'),
                        EqualizedConv2d(2*self.channels,self.channels,  padding=1, kernelSize=(3,3)),
                        self.f,
                        self.normalization_layer(),
                        EqualizedConv2d(self.channels,self.channels,  padding=1, kernelSize=(3,3)),
                        self.f,
                        self.normalization_layer(),
                        EqualizedConv2d(self.channels, 3, kernelSize=(1,1),  padding=0)
            ])
            self.introduce=nn.ModuleList([
                        nn.Upsample(scale_factor=(2,2), mode='nearest'),
                        EqualizedConv2d(2*self.channels, 3, kernelSize=(1,1),  padding=0)
            ])
        else:
            self.model=nn.ModuleList([
                        nn.Upsample(scale_factor=(2,2), mode='nearest'),
                        EqualizedConv2d(self.channels,self.channels,  padding=1, kernelSize=(3,3)),
                        self.f,
                        self.normalization_layer(),
                        EqualizedConv2d(self.channels,self.channels,  padding=1, kernelSize=(3,3)),
                        self.f,
                        self.normalization_layer(),
                        EqualizedConv2d(self.channels, 3, kernelSize=(1,1),  padding=0)
            ])
            self.introduce=nn.ModuleList([
                        nn.Upsample(scale_factor=(2,2), mode='nearest'),
                        EqualizedConv2d(self.channels, 3, kernelSize=(1,1),  padding=0)
            ])
        self.model.to(self.device)
        self.introduce.to(self.device)

    def increase_alpha(self, by=0.1):
        self.alpha=min(self.alpha+by, 1)

class Dis_residual(nn.Module):
    def __init__(self, res, channels, device='cuda',start_alpha=0.1, activation_f=nn.LeakyReLU(), normalize=True):
        super().__init__()
        self.alpha=start_alpha
        self.channels=channels
        self.f=activation_f
        self.device=device

    #    if normalize:
     #       self.normalization_layer=NormalizationLayer
      #  else:
       #     self.normalization_layer=nn.Identity
        #no normalization in descriminator

        if res>=64:
            self.model=nn.ModuleList([  # this is what stays on the pgan
                        EqualizedConv2d(3,self.channels,  padding=0, kernelSize=(1,1)), # from rgb
                        self.f,
                        EqualizedConv2d(self.channels,self.channels,  padding=1, kernelSize=(3,3)),
                        self.f,
                        EqualizedConv2d(self.channels,2*self.channels,  padding=1, kernelSize=(3,3)),
                        self.f,
                        nn.AvgPool2d(kernel_size=2, stride=2)
            ])
            self.introduce=nn.ModuleList([ # this is just for introducing new scale
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        EqualizedConv2d(3,2*self.channels,  padding=0, kernelSize=(1,1)) # from rgb
            ])
        else:
            self.model=nn.ModuleList([  # this is what stays on the pgan
                        EqualizedConv2d(3,self.channels,  padding=0, kernelSize=(1,1)), # from rgb
                        self.f,
                        EqualizedConv2d(self.channels,self.channels,  padding=1, kernelSize=(3,3)),
                        self.f,
                        EqualizedConv2d(self.channels,self.channels,  padding=1, kernelSize=(3,3)),
                        self.f,
                        nn.AvgPool2d(kernel_size=2, stride=2)
            ])
            self.introduce=nn.ModuleList([ # this is just for introducing new scale
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        EqualizedConv2d(3,self.channels,  padding=0, kernelSize=(1,1)) # from rgb
            ])
        self.model.to(self.device)
        self.introduce.to(self.device)

    def increase_alpha(self, by=0.1):
        self.alpha=min(self.alpha+by, 1)

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class miniBatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, subGroupSize=4):
        r"""
        Add a minibatch standard deviation channel to the current layer.
        In other words:
            1) Compute the standard deviation of the feature map over the minibatch
            2) Get the mean, over all pixels and all channels of thsi ValueError
            3) expand the layer and cocatenate it with the input

        Args:

            - x (tensor): previous layer
            - subGroupSize (int): size of the mini-batches on which the standard deviation
            should be computed
        """
        size = x.size()
        subGroupSize = min(size[0], subGroupSize)
        if size[0] % subGroupSize != 0:
            subGroupSize = size[0]
        G = int(size[0] / subGroupSize)
        if subGroupSize > 1:
            y = x.view(-1, subGroupSize, size[1], size[2], size[3])
            y = torch.var(y, 1)
            y = torch.sqrt(y + 1e-8)
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1)
            y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, subGroupSize, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3]))
        else:
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

        return torch.cat([x, y], dim=1)

class Generator(nn.Module):
    def __init__(self, latent_size=512, final_res=32, device='cuda', normalize=True, activation_f=nn.LeakyReLU()):
        super().__init__()
        #self.save_hyperparameters()
        self.curr_res=4
        self.final_res=final_res
        self.device=device
        self.alpha=0.0

        self.res_chanel_dict={
                            4:512,
                            8:512,
                            16:512,
                            32:512,
                            64:256,
                            128:128,
                            256:64,
                            512:32,
                            1024:16
                             }

        if normalize:
            self.normalization_layer=NormalizationLayer
        else:
            self.normalization_layer=nn.Identity

        self.f=activation_f
        self.layers=nn.ModuleList([
                    EqualizedConv2d(latent_size,latent_size,  padding=3, kernelSize=(4,4)),
                    self.f,
                    self.normalization_layer(),
                    EqualizedConv2d(latent_size,512,  padding=1, kernelSize=(3,3)),
                    self.f,
                    self.normalization_layer(),
                    EqualizedConv2d(512, 3, kernelSize=(1,1),  padding=0)
                    ])

        self.residual=None

    def forward(self, x):
        x = x.view(-1, num_flat_features(x), 1, 1).to(self.device)
      #  print(x.shape)
        for layer in self.layers[:-1]:
            x=layer(x)

        if self.residual:
            x_prim=torch.clone(x)
            for layer in self.residual.introduce:
                x_prim=layer(x_prim)
            for layer in self.residual.model:
                x=layer(x)
            x=self.alpha*x + (1-self.alpha)*x_prim
        else:
            x=self.layers[-1](x) #to rgb, if no upscaling is currently performed

        #normalize pixels into (0,1), same as CIFAR used
        return x



    def add_scale(self, start_alpha=0.1):
        self.curr_res*=2
        assert(0<=start_alpha<1)
        self.alpha=start_alpha

        self.residual=Gen_residual(self.curr_res, self.res_chanel_dict[self.curr_res], device=self.device,
                                 start_alpha=start_alpha, activation_f=self.f)

    def increase_alpha(self, by=0.1):
        assert(by>0)
        self.alpha=min(self.alpha+by, 1.0)

    def finish_adding_scale(self):
        self.layers= self.layers[:-1]+self.residual.model
        self.residual=None
        self.alpha=0.0

class Discriminator(nn.Module):
    def __init__(self, latent_size=512, final_res=32, device='cuda', normalize=True, activation_f=nn.LeakyReLU()):
        super().__init__()
        #self.save_hyperparameters()
        self.device=device
        self.curr_res=4
        self.final_res=final_res
        self.alpha=0.0
        self.res_chanel_dict={
                            4:512,
                            8:512,
                            16:512,
                            32:512,
                            64:256,
                            128:128,
                            256:64,
                            512:32,
                            1024:16
                             }
        if normalize:
            self.normalization_layer=NormalizationLayer
        else:
            self.normalization_layer=nn.Identity

        self.f=activation_f
        self.minibatch_layer=miniBatchStdDev()
        self.layers=nn.ModuleList([
                    EqualizedConv2d(3,512,  padding=0, kernelSize=(1,1)),
                    self.f,
                    self.minibatch_layer,
                    EqualizedConv2d(513,512,  padding=1, kernelSize=(3,3)),
                    self.f,
                    EqualizedConv2d(512,latent_size,  padding=0, kernelSize=(4,4)),
                    self.f,
                    ])
        self.decision_layer=nn.Sequential(EqualizedLinear(latent_size, 1))#, nn.Sigmoid())
        self.residual=None

    def forward(self, x, getFeature=False):

        #normalize both generated and training images
        x=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)

        if self.curr_res<x.shape[-1]:
            ratio=x.shape[-1]//self.curr_res
            x=nn.AvgPool2d(kernel_size=ratio, stride=ratio)(x)
        elif self.curr_res>x.shape[-1]:
            ratio=self.curr_res//x.shape[-1]
            x=nn.Upsample(scale_factor=(ratio,ratio), mode='nearest')(x)

        if self.residual:
            x_prim=torch.clone(x)
            for layer in self.residual.introduce:
                x_prim=layer(x_prim)
            for layer in self.residual.model:
                x=layer(x)
            x=self.alpha*x + (1-self.alpha)*x_prim
        else: # from rgb, if no upscaling is currently performed
            x=self.layers[0](x)

        for layer in self.layers[1:]:
            x=layer(x)

        x= x.view(-1, num_flat_features(x))
        out=self.decision_layer(x)
        return out


    def add_scale(self, start_alpha=0.1): # average pooling in descriminator
        self.curr_res*=2
        assert(0<=start_alpha<1)
        self.alpha=start_alpha

        self.residual=Dis_residual(self.curr_res, self.res_chanel_dict[self.curr_res], device= self.device,
                                 start_alpha=start_alpha,  activation_f=self.f)

    def increase_alpha(self, by=0.1):
        assert(by>0)
        self.alpha=min(self.alpha+by, 1.0)

    def finish_adding_scale(self):
        self.layers=self.residual.model+self.layers[1:]
        self.residual=None
        self.alpha=0


def WGANGP_loss(model, zi=None, xi=None, net=''):

    if net=='generator':
        gen_imgs=model.generator(zi)
        decisions=(model.discriminator(gen_imgs))

        g_loss=torch.mean(-decisions)
        return g_loss
    elif net=='discriminator':
        decisions_x=(model.discriminator(xi))
        gen_imgs=(model.generator(zi)).detach()
        decisions_z=(model.discriminator(gen_imgs))

        loss_x=-decisions_x
        loss_z=decisions_z

        lossEpsilon = (decisions_x ** 2) * 1e-3
        #loss_x += lossEpsilon

        if model.hparams.curr_res<xi.shape[-1]:
            ratio=xi.shape[-1]//model.hparams.curr_res
            xi=nn.AvgPool2d(kernel_size=ratio, stride=ratio)(xi)
        elif model.hparams.curr_res>xi.shape[-1]:
            ratio=model.hparams.curr_res//xi.shape[-1]
            xi=nn.Upsample(scale_factor=(ratio,ratio), mode='nearest')(xi)
        d_loss_grad_penalty = WGANGPGradientPenalty(xi,
                                            gen_imgs,
                                            model.discriminator,
                                            10.0,
                                            backward=False)

    #    print(loss_x.shape,loss_z.shape,lossEpsilon.shape,d_loss_grad_penalty.shape)

        d_loss=torch.mean(loss_x+loss_z+lossEpsilon+d_loss_grad_penalty)
        return d_loss

def original_loss(model, zi=None, xi=None, net=''):
    if net=='generator':
        gen_imgs=model.generator(zi)
        decisions=(model.discriminator(gen_imgs))

        g_loss=-torch.mean(torch.log(torch.sigmoid(decisions)))
        return g_loss
    elif net=='discriminator':
        decisions_x=(model.discriminator(xi))

        gen_imgs=(model.generator(zi)).detach()
        decisions_z=(model.discriminator(gen_imgs))

        loss_x=torch.log(torch.sigmoid(decisions_x))
        loss_z=torch.log(1-torch.sigmoid(decisions_z))
        d_loss=-torch.mean(loss_x+loss_z)  # ascend gradient, maximize. BUT torch optimis minimize loss, so return -mean
        return d_loss

def MSE_loss(model, zi=None, xi=None, net=''):
    #. D(x) represents the probability that x came from the data rather than pg.
    if net=='generator':
        gen_imgs=model.generator(zi)
        decisions=(model.discriminator(gen_imgs))

        no_samples=decisions.shape[0]
        reference = torch.zeros(no_samples, 1).to(model.device)
        return F.mse_loss(torch.sigmoid(decisions), reference)

    elif net=='discriminator':
        decisions_x=(model.discriminator(xi))

        gen_imgs=(model.generator(zi)).detach()
        decisions_z=(model.discriminator(gen_imgs))

        no_samples=decisions_z.shape[0]
        reference_x = torch.ones(no_samples, 1).to(model.device)
        reference_z = torch.zeros(no_samples, 1).to(model.device)
        loss_x = F.mse_loss(torch.sigmoid(decisions_x), reference_x)
        loss_z = F.mse_loss(torch.sigmoid(decisions_z), reference_z)
        return (loss_x+loss_z)
