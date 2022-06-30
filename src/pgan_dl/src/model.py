import pytorch_lightning as pl
import random

from .my_pgan import *


class PGAN(pl.LightningModule):
    def __init__(self, lr=0.1, latent_size=512, final_res=32, curr_res=4, k=1,
                 alpha=0.0, alpha_step=0.1, loss_f=WGANGP_loss,
                 normalize=True, activation_f=nn.LeakyReLU(negative_slope=0.2),
                 device='cuda', normalize_img=True):

        super().__init__()
        self.save_hyperparameters()

        self.id = f"{random.random():.3f}"[2:]
        self.loss_f = loss_f
        self.generator = Generator(latent_size=latent_size, final_res=final_res, normalize=normalize,
                                   activation_f=activation_f, device=device)
        self.discriminator = Discriminator(latent_size=latent_size, final_res=final_res, normalize=normalize,
                                           activation_f=activation_f, device=device, normalize_img=normalize_img)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        #  print(self.optimizers())

        xi, _ = batch
        zi = torch.randn(xi.shape[0], self.hparams.latent_size)  # TODO update zi sampling
        if self.hparams.normalize:
            zi = F.normalize(zi, dim=1, p=2)

        if optimizer_idx == 0:  # train Generator
            g_loss = self.loss_f(self, zi=zi, net='generator')

            self.log("generator_loss", g_loss)
            self.log("curr_res", float(self.hparams.curr_res))
            self.log("alpha", float(self.hparams.alpha))
            return g_loss

        if optimizer_idx > 0:  # train discriminator
            d_loss = self.loss_f(self, zi=zi, xi=xi, net='discriminator')
            self.log("discriminator_loss", d_loss)
            self.log("alpha", float(self.hparams.alpha))
            self.log("curr_res", float(self.hparams.curr_res))
            return d_loss

    def configure_optimizers(self):
        decay = 0
        if self.loss_f != WGANGP_loss:
            decay = 1e-4
        g_opt = torch.optim.Adam(self.generator.layers.parameters(), lr=self.hparams.lr, betas=(0, 0.99), eps=1e-8,
                                 weight_decay=decay)
        d_opt = torch.optim.Adam(self.discriminator.layers.parameters(), lr=self.hparams.lr, betas=(0, 0.99), eps=1e-8,
                                 weight_decay=decay)
        d_opt.add_param_group({'params': self.discriminator.decision_layer.parameters()})
        # should bne enough to load
        if self.generator.residual:
            g_opt.add_param_group({'params': self.generator.residual.model.parameters()})
            g_opt.add_param_group({'params': self.generator.residual.introduce.parameters()})
        if self.discriminator.residual:
            d_opt.add_param_group({'params': self.discriminator.residual.model.parameters()})
            d_opt.add_param_group({'params': self.discriminator.residual.introduce.parameters()})

        return [g_opt, d_opt]

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False,
    ):
        # update discrminator every step
        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)

        # update generator every k steps
        if optimizer_idx == 0:
            if (batch_idx + 1) % self.hparams.k == 0:
                # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                optimizer.step(closure=optimizer_closure)
            else:
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()

    def save_generated_images(self, n=10, save_dir='./images/'):
        t = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.ToPILImage()])  # , transforms.Resize(size=(256,256))])
        zi = torch.randn(n, self.hparams.latent_size)
        gen_imgs = self.generator(zi)
        i = 0
        for img in gen_imgs:
            t(img).save(save_dir + self.id + '_res_' + str(self.hparams.curr_res) + '_img_' + str(i) + '.png')
            i += 1

    def on_train_epoch_end(self):
        if self.current_epoch > int(
                0.5 / self.hparams.alpha_step) or self.hparams.curr_res > 4:  # first run X epochs on 4x4
            if self.hparams.alpha == 0 and self.hparams.curr_res < self.hparams.final_res:

                self.save_generated_images('data/07_model_output/')

                self.hparams.curr_res *= 2
                #  if self.hparams.curr_res==32:
                self.hparams.alpha_step *= 0.75

                self.generator.add_scale(start_alpha=self.hparams.alpha_step)
                self.discriminator.add_scale(start_alpha=self.hparams.alpha_step)
                self.hparams.alpha += self.hparams.alpha_step

                # update optimizers, 0-generator
                opts = self.optimizers()
                opts[0].add_param_group({'params': self.generator.residual.model.parameters()})
                opts[0].add_param_group({'params': self.generator.residual.introduce.parameters()})
                opts[1].add_param_group({'params': self.discriminator.residual.model.parameters()})
                opts[1].add_param_group({'params': self.discriminator.residual.introduce.parameters()})

            # print(model.generator.residual.model[1].module.weight.data[0,0])

            elif self.hparams.alpha >= 1:
                self.generator.finish_adding_scale()
                self.discriminator.finish_adding_scale()
                self.hparams.alpha = 0

                print("Done with: ", self.hparams.curr_res)
            elif self.hparams.alpha != 0:
                by = self.hparams.alpha_step
                self.generator.increase_alpha(by=by)
                self.discriminator.increase_alpha(by=by)
                self.hparams.alpha = min(self.hparams.alpha + by, 1.0)


class PGAN_loaded(PGAN):
    def __init__(self, lr=0.1, latent_size=512, final_res=32, curr_res=4, k=1,
                 alpha=0.0, alpha_step=0.1, loss_f=WGANGP_loss,
                 normalize=True, activation_f=nn.LeakyReLU(negative_slope=0.2),
                 device='cuda', normalize_img=True):

        super().__init__()
        self.save_hyperparameters(ignore=['activation_f', 'loss_f'])

        self.id = f"{random.random():.3f}"[2:]
        self.loss_f = loss_f
        self.generator = Generator(latent_size=latent_size, final_res=final_res, normalize=normalize,
                                   activation_f=activation_f, device=device)
        self.discriminator = Discriminator(latent_size=latent_size, final_res=final_res, normalize=normalize,
                                           activation_f=activation_f, device=device, normalize_img=normalize_img)

        resolution = 8
        self.generator.add_scale(start_alpha=self.hparams.alpha)
        self.discriminator.add_scale(start_alpha=self.hparams.alpha)
        while self.hparams.curr_res > resolution:
            self.generator.finish_adding_scale()
            self.discriminator.finish_adding_scale()
            # warnings.warn("ONLY FOR LOADING 8X8 MODEL WITH ALPHA!=1")
            self.generator.add_scale(start_alpha=self.hparams.alpha)
            self.discriminator.add_scale(start_alpha=self.hparams.alpha)
            resolution *= 2
        if self.hparams.alpha == 0:
            self.generator.finish_adding_scale()
