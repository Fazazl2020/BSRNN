#!/usr/bin/env python
# coding: utf-8

import os
import dataloader
import torch
import torch.nn.functional as F
import logging
from torchinfo import summary
from natsort import natsorted
import librosa
import numpy as np
from tqdm import tqdm
from module import *
from mbs_net import MBS_Net 

# ============================================
# CONFIGURATION - HARDCODED FOR SERVER
# ============================================
class Config:
    # Model selection: 'BSRNN', 'DB_Transform', 'MBS_Net', or 'MBS_Net'
    model_type = 'MBS_Net'  # Recommended: MBS_Net (memory-efficient)

    # Training hyperparameters
    epochs = 120
    batch_size = 6
    log_interval = 500
    decay_epoch = 10
    init_lr = 1e-3
    cut_len = int(16000 * 2)  # 2 seconds at 16kHz

    # Loss weights [RI, magnitude, phase, Metric Disc]
    # For MBS_Net: Use phase loss
    # For BSRNN/DB_Transform: phase weight = 0
    loss_weights = [0.3, 0.3, 0.4, 1.0]  # Added phase loss weight

    # MBS-Net specific
    use_pcs = False  # Use PCS during training (True) or only inference (False)
    pcs_alpha = 0.3  # PCS strength

    # Server paths - MODIFY THESE FOR YOUR SERVER
    data_dir = '/gdata/fewahab/data/VoicebanK-demand-16K'
    save_model_dir = '/ghome/fewahab/Sun-Models/Ab-6/M1/saved_model_mbsnet'  # Updated for MBS-Net

    # Resume training from checkpoint
    resume = False  # Set to True to resume from last checkpoint
    resume_path = None  # Will auto-detect latest checkpoint if None

    # Progress bar settings
    disable_progress_bar = False  # Set to True to reduce log verbosity

args = Config()
logging.basicConfig(level=logging.INFO)


# ============================================
# UTILITY FUNCTIONS
# ============================================
def check_nan_inf(tensor, name, raise_error=True):
    """Check for NaN/Inf and optionally log statistics"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        msg = f"⚠️  NaN/Inf detected in {name}!"
        msg += f"\n   NaN: {has_nan}, Inf: {has_inf}"
        msg += f"\n   Min: {tensor[~torch.isnan(tensor)].min() if not has_nan else 'NaN'}"
        msg += f"\n   Max: {tensor[~torch.isinf(tensor)].max() if not has_inf else 'Inf'}"
        msg += f"\n   Shape: {tensor.shape}"
        logging.error(msg)
        if raise_error:
            raise ValueError(f"NaN/Inf in {name}")
    return has_nan or has_inf


def save_checkpoint(epoch, model, discriminator, optimizer, optimizer_disc,
                   scheduler_G, scheduler_D, gen_loss, save_dir):
    """Save complete training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_disc_state_dict': optimizer_disc.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
        'gen_loss': gen_loss,
        'config': {
            'model_type': args.model_type,
            'init_lr': args.init_lr,
            'batch_size': args.batch_size,
        }
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save with epoch number
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)

    # Also save as latest
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

    logging.info(f"✓ Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, discriminator, optimizer,
                   optimizer_disc, scheduler_G, scheduler_D):
    """Load checkpoint and restore training state"""
    if not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0

    logging.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
    scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
    scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    logging.info(f"✓ Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")
    logging.info(f"   Previous loss: {checkpoint['gen_loss']:.4f}")

    return start_epoch


def find_latest_checkpoint(save_dir):
    """Find the latest checkpoint in directory"""
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    if os.path.exists(latest_path):
        return latest_path

    # Fallback: find highest epoch number
    if not os.path.exists(save_dir):
        return None

    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None

    # Extract epoch numbers and find max
    epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    return os.path.join(save_dir, f'checkpoint_epoch_{latest_epoch}.pt')


class Trainer:
    def __init__(self, train_ds, test_ds):
        self.n_fft = 512
        self.hop = 128
        self.train_ds = train_ds
        self.test_ds = test_ds

        # Model selection based on config
        if args.model_type == 'MBS_Net':
            self.model = MBS_Net(
                num_channel=128,
                num_layers=4,
                num_bands=30,
                d_state=12,
                chunk_size=32
            ).cuda()
            logging.info("Using MBS-Net Optimized (memory-efficient, ~2.3M params)")
        elif args.model_type == 'MBS_Net':
            self.model = MBS_Net_Original(num_channel=128, num_layers=4).cuda()
            logging.info("Using MBS-Net architecture (Mamba + Explicit Phase)")
        elif args.model_type == 'DB_Transform':
            self.model = DB_Transform(num_channel=128, num_heads=4).cuda()
            logging.info("Using DB-Transform architecture")
        elif args.model_type == 'BSRNN':
            self.model = BSRNN(num_channel=64, num_layer=5).cuda()
            logging.info("Using BSRNN baseline")
        else:
            raise ValueError(f"Unknown model_type: {args.model_type}")

        # Count and display parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: Total={total_params/1e6:.2f}M, Trainable={trainable_params/1e6:.2f}M")

        self.discriminator = Discriminator(ndf=16).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=args.init_lr)

    def compute_phase_loss(self, est_spec, clean_spec):
        """
        Compute wrapped phase loss for explicit phase modeling.

        Args:
            est_spec: Estimated complex spectrogram
            clean_spec: Clean complex spectrogram
        Returns:
            phase_loss: L1 loss on wrapped phase difference
        """
        # NUMERICAL STABILITY: Add small epsilon to prevent angle() instability
        # when magnitude is near zero
        eps = 1e-8
        est_spec_stable = est_spec + eps
        clean_spec_stable = clean_spec + eps

        est_phase = torch.angle(est_spec_stable)
        clean_phase = torch.angle(clean_spec_stable)

        # Wrap phase difference to [-π, π]
        phase_diff = torch.remainder(est_phase - clean_phase + np.pi, 2*np.pi) - np.pi

        # L1 loss on wrapped difference
        phase_loss = F.l1_loss(phase_diff, torch.zeros_like(phase_diff))

        # Check for NaN
        check_nan_inf(phase_loss, "phase_loss", raise_error=True)

        return phase_loss

    def train_step(self, batch, use_disc):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        self.optimizer.zero_grad()
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)

        est_spec = self.model(noisy_spec)

        # CRITICAL: Check for NaN immediately after model forward
        check_nan_inf(est_spec, "est_spec", raise_error=True)

        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** (0.3)
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** (0.3)
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** (0.3)

        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec,clean_spec)

        # Check losses
        check_nan_inf(loss_mag, "loss_mag", raise_error=True)
        check_nan_inf(loss_ri, "loss_ri", raise_error=True)

        # Add phase loss for MBS_Net variants (explicit phase modeling)
        if args.model_type in ['MBS_Net', 'MBS_Net']:
            loss_phase = self.compute_phase_loss(est_spec, clean_spec)
        else:
            loss_phase = torch.tensor(0.0).cuda()

        if use_disc is False:
            # loss_weights = [RI, magnitude, phase, Metric Disc]
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * loss_phase
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * loss_phase + args.loss_weights[3] * gen_loss_GAN

        # Final NaN check before backward
        check_nan_inf(loss, "total_loss", raise_error=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
        self.optimizer.step()

        est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                           onesided =True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        # Store PESQ score for logging (denormalize from [0,1] back to [-0.5, 4.5] range)
        pesq_raw = None
        if pesq_score is not None:
            pesq_raw = (pesq_score.mean().item() * 5) - 0.5

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None and pesq_score_n is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels.float()) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)

            discrim_loss_metric.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5)
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item(), pesq_raw

    @torch.no_grad()
    def test_step(self, batch,use_disc):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                                onesided=True,return_complex=True)

        est_spec = self.model(noisy_spec)
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** (0.3)
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** (0.3)
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** (0.3)

        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        # Add phase loss for MBS_Net
        if args.model_type == 'MBS_Net':
            loss_phase = self.compute_phase_loss(est_spec, clean_spec)
        else:
            loss_phase = torch.tensor(0.0).cuda()

        if use_disc is False:
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * loss_phase
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * loss_phase + args.loss_weights[3] * gen_loss_GAN

        est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=torch.hann_window(self.n_fft).cuda(),
                           onesided =True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        # Store PESQ score for logging (denormalize from [0,1] back to [-0.5, 4.5] range)
        pesq_raw = None
        if pesq_score is not None:
            pesq_raw = (pesq_score.mean().item() * 5) - 0.5

        if pesq_score is not None and pesq_score_n is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item(), pesq_raw

    def test(self,use_disc):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        pesq_total = 0.
        pesq_count = 0
        # Use disable_progress_bar to control verbosity
        for idx, batch in enumerate(tqdm(self.test_ds, disable=args.disable_progress_bar)):
            step = idx + 1
            loss, disc_loss, pesq_raw = self.test_step(batch,use_disc)
            gen_loss_total += loss
            disc_loss_total += disc_loss
            if pesq_raw is not None:
                pesq_total += pesq_raw
                pesq_count += 1
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        pesq_avg = pesq_total / pesq_count if pesq_count > 0 else 0

        template = 'TEST - Generator loss: {:.4f}, Discriminator loss: {:.4f}, PESQ: {:.4f}'
        logging.info(template.format(gen_loss_avg, disc_loss_avg, pesq_avg))

        return gen_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_epoch, gamma=0.98)
        scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=args.decay_epoch, gamma=0.98)

        # Resume from checkpoint if requested
        start_epoch = 0
        if args.resume:
            checkpoint_path = args.resume_path
            if checkpoint_path is None:
                checkpoint_path = find_latest_checkpoint(args.save_model_dir)

            if checkpoint_path:
                start_epoch = load_checkpoint(
                    checkpoint_path, self.model, self.discriminator,
                    self.optimizer, self.optimizer_disc,
                    scheduler_G, scheduler_D
                )
            else:
                logging.warning("No checkpoint found, starting from scratch")

        for epoch in range(start_epoch, args.epochs):
            self.model.train()
            self.discriminator.train()

            loss_total = 0
            loss_gan = 0
            pesq_total = 0
            pesq_count = 0

            if epoch >= (args.epochs/2):
                use_disc = True
            else:
                use_disc = False

            # Use disable_progress_bar to control verbosity
            for idx, batch in enumerate(tqdm(self.train_ds, disable=args.disable_progress_bar)):
                step = idx + 1
                try:
                    loss, disc_loss, pesq_raw = self.train_step(batch,use_disc)
                except ValueError as e:
                    if "NaN" in str(e):
                        logging.error(f"⚠️  Training stopped due to NaN at Epoch {epoch}, Step {step}")
                        logging.error(f"   Saving emergency checkpoint before exit...")
                        save_checkpoint(epoch, self.model, self.discriminator,
                                      self.optimizer, self.optimizer_disc,
                                      scheduler_G, scheduler_D, 999.0,
                                      args.save_model_dir)
                        raise
                    else:
                        raise

                loss_total = loss_total + loss
                loss_gan = loss_gan + disc_loss
                if pesq_raw is not None:
                    pesq_total += pesq_raw
                    pesq_count += 1

                if (step % args.log_interval) == 0:
                    pesq_avg = pesq_total/pesq_count if pesq_count > 0 else 0
                    template = 'Epoch {}, Step {}, loss: {:.4f}, disc_loss: {:.4f}, PESQ: {:.4f}'
                    logging.info(template.format(epoch, step, loss_total/step, loss_gan/step, pesq_avg))

            gen_loss = self.test(use_disc)

            # Save full checkpoint (NEW!)
            save_checkpoint(epoch, self.model, self.discriminator,
                          self.optimizer, self.optimizer_disc,
                          scheduler_G, scheduler_D, gen_loss,
                          args.save_model_dir)

            # Also save old-style checkpoints for compatibility
            path = os.path.join(args.save_model_dir, 'gene_epoch_' + str(epoch) + '_' + str(gen_loss)[:5])
            path_d = os.path.join(args.save_model_dir, 'disc_epoch_' + str(epoch))
            torch.save(self.model.state_dict(), path)
            torch.save(self.discriminator.state_dict(), path_d)

            scheduler_G.step()
            scheduler_D.step()

def main():
    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 4, args.cut_len)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()

if __name__ == '__main__':
    main()
