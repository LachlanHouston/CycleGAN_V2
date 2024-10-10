from CycleGAN.models.generator import Generator
from CycleGAN.models.discriminator import Discriminator
import pytorch_lightning as L
from pytorch_lightning import Trainer
import torch
import numpy as np
import trimesh
import random
import wandb


# define the Autoencoder class containing the training setup
class Autoencoder(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            print(f"Setting {k} to {v}")
            setattr(self, k, v)

        # define the discriminator and generator
        self.m_discriminator=Discriminator().to(self.device)
        self.m_generator=Generator().to(self.device)

        self.f_discriminator=Discriminator().to(self.device)
        self.f_generator=Generator().to(self.device)

        self.g_learning_rate = 1e-4
        self.d_learning_rate = 1e-4

        self.mse = torch.nn.MSELoss()

        self.custom_global_step = 0
        self.save_hyperparameters() # save hyperparameters to Weights and Biases
        self.automatic_optimization = False

    def forward(self, batch):
        real_male = batch[0].to(self.device)
        real_female = batch[1].to(self.device)
        return self.m_generator(real_female), self.f_generator(real_male)
    
    def chamfer_loss(self, x, y):
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        mins, _ = torch.min(P, 1)                       # find the nearest gt point for each pred point
        loss_1 = torch.mean(mins)                       # take mean across got <-> pred loss
        mins, _ = torch.min(P, 2)                       # find the nearest pred point for each gt point
        loss_2 = torch.mean(mins)                       # take mean across pred <-> gt loss
        return (loss_1 + loss_2)
    
    def generator_loss(self, fake_male, fake_female, real_male, real_female, test_logger=False):
        D_M_fake = self.m_discriminator(fake_male)
        D_F_fake = self.f_discriminator(fake_female)

        # Losses for generator
        loss_G_M = - D_M_fake.mean()
        #adversarial loss for female
        loss_G_F = - D_F_fake.mean()

        # generate cycle-female and cycle-male
        cycle_female = self.f_generator(fake_male)
        cycle_male = self.m_generator(fake_female)

        # compute chamfer loss between real and cycle point clouds
        cycle_female_loss = self.chamfer_loss(cycle_female, real_female)
        cycle_male_loss = self.chamfer_loss(cycle_male, real_male)

        # compute cycle-consistency loss
        cycle_loss = (cycle_female_loss + cycle_male_loss) * 10

        # add all generator losses together to obtain full generator loss
        G_loss = (loss_G_F + loss_G_M + cycle_loss)

        if test_logger == True:
            self.log('Generator Loss', G_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('Cycle Loss', cycle_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('G Adversarial Loss', loss_G_F + loss_G_M, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return G_loss
    
    def gradient_penalty(self, real_male, real_female, fake_male, fake_female, discriminator_m, discriminator_f):
        grad_discriminator_m = discriminator_m
        grad_discriminator_f = discriminator_f
        # create interpolated samples
        alpha = torch.rand(real_male.shape[0], 1, 1, device=self.device)
        differences = fake_male - real_male + fake_female - real_female
        interpolates = (real_male + real_female) + (alpha * differences)
        interpolates.requires_grad_(True)

        # calculate the output of the discriminator for the interpolated samples and compute the gradients
        interpolates_m = grad_discriminator_m(interpolates) # B x 1
        interpolates_f = grad_discriminator_f(interpolates) # B x 1
        ones = torch.ones(interpolates_m.size(), device=self.device) # B x 1
        ones = torch.ones(interpolates_f.size(), device=self.device) # B x 1
        gradients_m = torch.autograd.grad(outputs=interpolates_m, inputs=interpolates, grad_outputs=ones, 
                                        create_graph=True, retain_graph=True)[0] # B x C x H x W
        gradients_f = torch.autograd.grad(outputs=interpolates_f, inputs=interpolates, grad_outputs=ones, 
                                        create_graph=True, retain_graph=True)[0] # B x C x H x W

        # calculate the combined gradient penalty
        gradients = torch.cat([gradients_m, gradients_f], dim=0)
        gradients = gradients.view(real_male.shape[0], -1)
        grad_norms = gradients.norm(2, dim=1)
        gradient_penalty = ((grad_norms - 1) ** 2).mean()
        return gradient_penalty * 10
    
    def discriminator_loss(self, real_male, real_female, fake_male, fake_female, D_M_real, D_M_fake, D_F_real, D_F_fake, test_logger=False, train_logger=False):
        # Losses for discriminator
        D_M_loss = D_M_fake.mean() - D_M_real.mean()
        D_F_loss = D_F_fake.mean() - D_F_real.mean()

        gradient_penalty_M = self.gradient_penalty(real_male, real_female, fake_male, fake_female, self.m_discriminator, self.f_discriminator)

        D_loss = (D_M_loss + D_F_loss + gradient_penalty_M) / 3

        if train_logger == True:
            self.log('Gradient Penalty M', gradient_penalty_M, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            #self.log('Gradient Penalty F', gradient_penalty_F, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if test_logger == True:
            self.log('Discriminator Loss M', D_M_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('Discriminator Loss F', D_F_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('Discriminator Loss', D_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return D_loss
    
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(list(self.m_generator.parameters()) + list(self.f_generator.parameters()), lr=self.g_learning_rate)
        d_opt = torch.optim.Adam(list(self.m_discriminator.parameters()) + list(self.f_discriminator.parameters()), lr=self.d_learning_rate)
        return g_opt, d_opt

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        
        # unpack batched data
        real_male = batch[0].float().to(self.device)
        real_female = batch[1].float().to(self.device)
        
        self.toggle_optimizer(g_opt)
        # generate fake samples and discriminator predictions
        ## female -> male
        fake_male = self.m_generator(real_female)
        ## male -> female
        fake_female = self.f_generator(real_male)
        # Losses for generator
        G_loss = self.generator_loss(fake_male, fake_female, real_male, real_female) 

        # update generator
        self.manual_backward(G_loss)
        g_opt.step()
        g_opt.zero_grad()
        self.untoggle_optimizer(g_opt)

        self.toggle_optimizer(d_opt)
        fake_male = self.m_generator(real_female).detach()
        fake_female = self.f_generator(real_male).detach()
        D_M_real = self.m_discriminator(real_male)
        D_M_fake = self.m_discriminator(fake_male)
        D_F_real = self.f_discriminator(real_female)
        D_F_fake = self.f_discriminator(fake_female)

        D_loss = self.discriminator_loss(real_male, real_female, fake_male, fake_female, D_M_real, D_M_fake, D_F_real, D_F_fake, train_logger=True)

        # update discriminator
        self.manual_backward(D_loss)
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)
        
        self.custom_global_step += 1

    def validation_step(self, batch, batch_idx):
        visualise_idx = random.randint(0, batch[0].shape[0]-1)

        with torch.enable_grad():
            # unpack batched data
            real_male = batch[0].float().to(self.device)
            real_female = batch[1].float().to(self.device)

            # generate fake samples and discriminator predictions
            ## female -> male
            fake_male = self.m_generator(real_female)
            D_M_real = self.m_discriminator(real_male)
            D_M_fake = self.m_discriminator(fake_male)

            ## male -> female
            fake_female = self.f_generator(real_male)
            D_F_real = self.m_discriminator(real_female)
            D_F_fake = self.m_discriminator(fake_male)

            # losses for discriminator
            D_loss = self.discriminator_loss(real_male, real_female, fake_male, fake_female, D_M_real, D_M_fake, D_F_real, D_F_fake, test_logger=True)

            # losses for generator
            G_loss = self.generator_loss(fake_male, fake_female, real_male, real_female, test_logger=True) 
            
        if self.logging is True:
            if batch_idx == 0:
                self.logger.experiment.log({"OG_male": wandb.Object3D(real_male[visualise_idx].detach().cpu().numpy())})
                self.logger.experiment.log({"fake female": wandb.Object3D(fake_female[visualise_idx].detach().cpu().numpy())})
                
        else:
            print(D_loss, G_loss)
        


if __name__ == "__main__":
    fake_male_data = torch.rand(2, 2, 2048, 3).requires_grad_(True)
    fake_female_data = torch.rand(2, 2, 2048, 3).requires_grad_(True)
    fake_train_batch = torch.stack([fake_male_data, fake_female_data], dim=0)
    fake_test_batch = torch.stack([fake_male_data, fake_female_data], dim=0)

    model = Autoencoder(logging=False)

    # define the pytorch lightning trainer 
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=5,
        check_val_every_n_epoch=1,
        deterministic=True,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
    )
    
    # train the model
    print("Starting new training")
    trainer.fit(model, fake_train_batch, fake_test_batch)
