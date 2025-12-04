#!/usr/bin/env python
# coding: utf-8
"""
Training script for Ablation 2: Dual-Path BiMamba + Uniform Decoder

Complete standalone training script.
Expected PESQ: 3.1-3.2
Parameters: ~3.96M
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from natsort import natsorted
import librosa
import numpy as np

# Add parent directory to import dataloader and module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import dataloader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Baseline'))
from module import Discriminator, batch_pesq

# Import model from THIS directory
from mbs_net import MBS_Net

# ==================== TRAINING CONFIGURATION ====================
class TrainingConfig:
    """Easy-to-edit training configuration"""

    # Training Hyperparameters
    batch_size = 6
    epochs = 120
    init_lr = 1e-3
    decay_epoch = 10

    # Resume Training
    resume_checkpoint = None
    resume_from_best = False

    # Data - EDIT THESE FOR YOUR SERVER
    data_dir = '/gdata/fewahab/data/VoicebanK-demand-16K'  # Change this to your data path
    save_model_dir = '/ghome/fewahab/Sun-Models/Ab-6/M2/saved_model'  # Saves in current directory

    # Loss Weights
    loss_weights = [0.5, 0.5, 1]

    # Logging
    log_interval = 100
    cut_len = int(16000 * 2)
# ================================================================

logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, train_ds, test_ds):
        self.n_fft = 512
        self.hop = 128
        self.train_ds = train_ds
        self.test_ds = test_ds

        # Create model - Ablation 2 (MEMORY-OPTIMIZED)
        self.model = MBS_Net(
            num_channel=128,
            num_layers=2,  # Reduced from 4 for memory
            num_bands=30,
            d_state=12,    # Reduced from 16 for memory
            chunk_size=32
        ).cuda()
        logging.info("Ablation 2: Dual-Path BiMamba + Uniform Decoder (MEMORY-OPTIMIZED)")

        # Log parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: Total={total_params/1e6:.2f}M, Trainable={trainable_params/1e6:.2f}M")
        logging.info(f"Expected: ~3.7M params")

        self.discriminator = Discriminator(ndf=16).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=TrainingConfig.init_lr)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=TrainingConfig.init_lr)

        self.start_epoch = 0
        self.best_loss = float('inf')

        # Resume from checkpoint if specified
        best_path = os.path.join(TrainingConfig.save_model_dir, 'best_model.pth')
        resume_path = TrainingConfig.resume_checkpoint

        if TrainingConfig.resume_from_best and os.path.exists(best_path):
            logging.info("Resuming from best model...")
            self.load_checkpoint(best_path)
        elif resume_path is not None and os.path.exists(resume_path):
            logging.info(f"Resuming from checkpoint: {resume_path}")
            self.load_checkpoint(resume_path)

    def save_checkpoint(self, epoch, gen_loss, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
            'best_loss': self.best_loss,
            'gen_loss': gen_loss
        }

        if not os.path.exists(TrainingConfig.save_model_dir):
            os.makedirs(TrainingConfig.save_model_dir)

        latest_path = os.path.join(TrainingConfig.save_model_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(TrainingConfig.save_model_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logging.info(f"✓ New best model saved! Loss: {gen_loss:.6f}")

        if (epoch + 1) % 5 == 0:
            epoch_ckpt = os.path.join(TrainingConfig.save_model_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_ckpt)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.start_epoch = checkpoint['epoch'] + 1

        logging.info(f"✓ Resumed from epoch {checkpoint['epoch']}")

    def train_step(self, batch, use_disc):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        self.optimizer.zero_grad()
        noisy_spec = torch.stft(
            noisy, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True, return_complex=True
        )
        clean_spec = torch.stft(
            clean, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True, return_complex=True
        )

        est_spec = self.model(noisy_spec)
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** 0.3
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** 0.3
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** 0.3

        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        if not use_disc:
            loss = (
                TrainingConfig.loss_weights[0] * loss_ri +
                TrainingConfig.loss_weights[1] * loss_mag
            )
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = (
                TrainingConfig.loss_weights[0] * loss_ri +
                TrainingConfig.loss_weights[1] * loss_mag +
                TrainingConfig.loss_weights[2] * gen_loss_GAN
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
        self.optimizer.step()

        est_audio = torch.istft(
            est_spec, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True
        )

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        pesq_raw = None
        if pesq_score is not None:
            pesq_raw = (pesq_score.mean().item() * 5) - 0.5

        if pesq_score is not None and pesq_score_n is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)
            discrim_loss_metric = (
                F.mse_loss(predict_max_metric.flatten(), one_labels.float()) +
                F.mse_loss(predict_enhance_metric.flatten(), pesq_score) +
                F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
            )

            discrim_loss_metric.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5)
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item(), pesq_raw

    @torch.no_grad()
    def test_step(self, batch, use_disc):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        noisy_spec = torch.stft(
            noisy, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True, return_complex=True
        )
        clean_spec = torch.stft(
            clean, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True, return_complex=True
        )

        est_spec = self.model(noisy_spec)
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** 0.3
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** 0.3
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** 0.3

        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        if not use_disc:
            loss = (
                TrainingConfig.loss_weights[0] * loss_ri +
                TrainingConfig.loss_weights[1] * loss_mag
            )
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = (
                TrainingConfig.loss_weights[0] * loss_ri +
                TrainingConfig.loss_weights[1] * loss_mag +
                TrainingConfig.loss_weights[2] * gen_loss_GAN
            )

        est_audio = torch.istft(
            est_spec, self.n_fft, self.hop,
            window=torch.hann_window(self.n_fft).cuda(),
            onesided=True
        )

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        pesq_raw = None
        if pesq_score is not None:
            pesq_raw = (pesq_score.mean().item() * 5) - 0.5

        if pesq_score is not None and pesq_score_n is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)
            discrim_loss_metric = (
                F.mse_loss(predict_max_metric.flatten(), one_labels) +
                F.mse_loss(predict_enhance_metric.flatten(), pesq_score) +
                F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
            )
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item(), pesq_raw

    def test(self, use_disc):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        pesq_total = 0.
        pesq_count = 0

        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss, pesq_raw = self.test_step(batch, use_disc)
            gen_loss_total += loss
            disc_loss_total += disc_loss

            if pesq_raw is not None:
                pesq_total += pesq_raw
                pesq_count += 1

        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        pesq_avg = pesq_total / pesq_count if pesq_count > 0 else 0.0

        template = 'TEST - Generator loss: {:.4f}, Discriminator loss: {:.4f}, PESQ: {:.4f}'
        logging.info(template.format(gen_loss_avg, disc_loss_avg, pesq_avg))

        return gen_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=TrainingConfig.decay_epoch,
            gamma=0.98,
            last_epoch=self.start_epoch - 1
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc,
            step_size=TrainingConfig.decay_epoch,
            gamma=0.98,
            last_epoch=self.start_epoch - 1
        )

        for epoch in range(self.start_epoch, TrainingConfig.epochs):
            self.model.train()
            self.discriminator.train()

            loss_total = 0.0
            loss_gan = 0.0
            pesq_total = 0.0
            pesq_count = 0

            use_disc = epoch >= (TrainingConfig.epochs / 2)

            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss, pesq_raw = self.train_step(batch, use_disc)

                loss_total += loss
                loss_gan += disc_loss
                if pesq_raw is not None:
                    pesq_total += pesq_raw
                    pesq_count += 1

                if (step % TrainingConfig.log_interval) == 0:
                    pesq_avg = pesq_total / pesq_count if pesq_count > 0 else 0.0
                    template = 'Epoch {}, Step {}, loss: {:.4f}, disc_loss: {:.4f}, PESQ: {:.4f}'
                    logging.info(
                        template.format(
                            epoch, step,
                            loss_total / step,
                            loss_gan / step,
                            pesq_avg
                        )
                    )

            gen_loss = self.test(use_disc)

            is_best = gen_loss < self.best_loss
            if is_best:
                self.best_loss = gen_loss

            self.save_checkpoint(epoch, gen_loss, is_best=is_best)

            path = os.path.join(
                TrainingConfig.save_model_dir,
                'gene_epoch_' + str(epoch) + '_' + str(gen_loss)[:5]
            )
            path_d = os.path.join(
                TrainingConfig.save_model_dir,
                'disc_epoch_' + str(epoch)
            )
            if not os.path.exists(TrainingConfig.save_model_dir):
                os.makedirs(TrainingConfig.save_model_dir)
            torch.save(self.model.state_dict(), path)
            torch.save(self.discriminator.state_dict(), path_d)

            scheduler_G.step()
            scheduler_D.step()


def main():
    print("=" * 70)
    print("ABLATION 2: Dual-Path BiMamba + Uniform Decoder")
    print("=" * 70)
    print(f"Expected PESQ: 3.1-3.2")
    print(f"Expected Parameters: ~4.42M")
    print("=" * 70)

    logging.info("Training configuration:")
    logging.info(
        f"epochs={TrainingConfig.epochs}, batch_size={TrainingConfig.batch_size}, "
        f"init_lr={TrainingConfig.init_lr}, data_dir={TrainingConfig.data_dir}, "
        f"save_model_dir={TrainingConfig.save_model_dir}"
    )
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    logging.info(f"Available GPUs: {available_gpus}")

    train_ds, test_ds = dataloader.load_data(
        TrainingConfig.data_dir,
        TrainingConfig.batch_size,
        4,
        TrainingConfig.cut_len
    )
    trainer = Trainer(train_ds, test_ds)
    trainer.train()


if __name__ == '__main__':
    main()
