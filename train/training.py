from configs.config import Config
import torch
from torch import nn
from models.classifier import Classifier
from models.discriminator import Discriminator
from models.unet import UNet
from tqdm.auto import tqdm
from utils.image_utils import show_tensor_images


class Trainer:
    def __init__(self):
        self.classifier_opt = None
        self.classifier = None
        self.disc_opt = None
        self.disc = None
        self.gen_opt = None
        self.gen = None
        self.adv_criterion = nn.BCEWithLogitsLoss().to(Config.device)
        self.classifier_criterion = nn.BCEWithLogitsLoss().to(Config.device)
        self.recon_criterion = nn.L1Loss().to(Config.device)

    def __adjust_learning_rate(self, optimizer, global_step, base_lr, lr_decay_rate=0.1, lr_decay_steps=6e4):
        """Adjust the learning rate of the params of an optimizer."""
        lr = base_lr * (lr_decay_rate ** (global_step / lr_decay_steps))
        if lr < 1e-6:
            lr = 1e-6

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def initialize(self):
        self.gen = UNet(input_channels=Config.input_dim, output_channels=Config.real_dim, hidden_channels=32).to(Config.device)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=Config.base_lr_gen, betas=(Config.beta1, Config.beta2))
        self.disc = Discriminator(Config.input_dim + Config.real_dim).to(Config.device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=Config.base_lr_disc, betas=(Config.beta1, Config.beta2))
        self.classifier = Classifier(Config.input_dim + Config.real_dim).to(Config.device)
        self.classifier_opt = torch.optim.Adam(self.classifier.parameters(), lr=Config.base_lr_classifier, betas=(Config.beta1, Config.beta2))
        self.__weights_init(self.gen)
        self.__weights_init(self.disc)
        self.__weights_init(self.classifier)


    def __weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0.0)

    def train(self, n_epochs,dataloader):
        self.gen.train()
        self.disc.train()
        self.classifier.train()

        mean_generator_loss = 0
        mean_discriminator_loss = 0
        mean_classifier_loss = 0

        start_epoch = 0
        cur_step = 0

        for epoch in range(start_epoch, n_epochs):
            count = 0
            for image, _ in tqdm(dataloader):
                image = image.to(Config.device)
                real = image[:, 1:, :, :]  # (N,C,H,W)
                condition = image[:, 0:1, :, :]
                if count != 0 and count != 1:
                    raise Exception("Error, count value is not valid")
                if count == 0:
                    self.gen.switch = 0
                elif count == 1:
                    self.gen.switch = 1

                # update generator
                self.__adjust_learning_rate(self.gen_opt,
                                     cur_step,
                                     base_lr=Config.base_lr_gen,
                                     lr_decay_rate=Config.lr_decay_rate,
                                     lr_decay_steps=Config.lr_decay_steps)

                self.gen_opt.zero_grad()
                fake = self.gen(condition)
                fake_img_lab = torch.cat([condition, fake], dim=1).to(Config.device)
                disc_fake_hat = self.disc(fake_img_lab)
                gen_adv_loss = self.adv_criterion(disc_fake_hat, torch.full_like(disc_fake_hat, 0.9))
                gen_rec_loss = self.recon_criterion(image, fake_img_lab)
                tt = self.classifier(fake_img_lab)
                disc_classification_loss = self.classifier_criterion(tt, torch.full_like(tt, 0.01 if count == 0 else 0.99))
                gen_loss = gen_adv_loss + Config.lambda_recon * gen_rec_loss + Config.beta * disc_classification_loss
                gen_loss.backward()
                self.gen_opt.step()

                # update discriminator
                self.__adjust_learning_rate(self.disc_opt,
                                     cur_step,
                                     base_lr=Config.base_lr_disc,
                                     lr_decay_rate=Config.lr_decay_rate,
                                     lr_decay_steps=Config.lr_decay_steps)
                self.disc_opt.zero_grad()
                disc_real_hat = self.disc(image)
                disc_real_loss = self.adv_criterion(disc_real_hat, torch.full_like(disc_real_hat, 0.9))
                disc_fake_hat = self.disc(fake_img_lab.detach())  # Detach generator
                disc_fake_loss = self.adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
                disc_loss = (disc_fake_loss + disc_real_loss)
                disc_loss.backward()
                self.disc_opt.step()

                # update classifier
                self.__adjust_learning_rate(self.classifier_opt,
                                     cur_step,
                                     base_lr=Config.base_lr_classifier,
                                     lr_decay_rate=Config.lr_decay_rate,
                                     lr_decay_steps=Config.lr_decay_steps)

                self.classifier_opt.zero_grad()
                tt = self.classifier(fake_img_lab.detach())
                disc_classification_loss = self.classifier_criterion(tt, torch.full_like(tt, 0.01 if count == 0 else 0.99))
                disc_classification_loss.backward()  # Update gradients
                self.classifier_opt.step()

                count += 1
                count = count % 2

                mean_discriminator_loss = (cur_step * mean_discriminator_loss + disc_loss.item()) / (cur_step + 1)
                mean_generator_loss = (cur_step * mean_generator_loss + gen_loss.item()) / (cur_step + 1)
                mean_classifier_loss = (cur_step * mean_classifier_loss + disc_classification_loss.item()) / (
                            cur_step + 1)

                ### Visualization ###
                if cur_step % Config.display_step == 0:
                    if cur_step > 0:
                        print(
                            f"Epoch {epoch}: Step {cur_step}: Classifier loss: ,{disc_classification_loss.item()}, Generator (U-Net) loss: {gen_loss.item()}, Discriminator loss: {disc_loss.item()}")
                        print(
                            f"Epoch {epoch}: Step {cur_step}: Classifier loss mean: ,{mean_classifier_loss}, Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss mean: {mean_discriminator_loss}")
                    else:
                        print("Pretrained initial state")

                    show_tensor_images((condition + 1.) * 50.,
                                       size=(3, Config.target_shape, Config.target_shape),
                                       gray_scale=True)
                    show_tensor_images(torch.cat([(condition + 1.) * 50., real * 110], 1),
                                       size=(Config.real_dim + Config.input_dim, Config.target_shape, Config.target_shape))
                    show_tensor_images(torch.cat([(condition + 1.) * 50., fake * 110], 1),
                                       size=(Config.real_dim + Config.input_dim, Config.target_shape, Config.target_shape))
                cur_step += 1

