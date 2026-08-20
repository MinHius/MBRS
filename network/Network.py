from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator
from kornia.losses import ssim_loss


import torch
import torch.nn as nn
import kornia
from kornia.losses import ssim_loss

class Network:
    def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, 
                 total_epochs, with_diffusion=False, only_decoder=False):
        # device
        self.device = device

        # network
        if not with_diffusion:
            self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers).to(device)
        else:
            self.encoder_decoder = EncoderDecoder_Diffusion(H, W, message_length, noise_layers).to(device)

        self.discriminator = Discriminator().to(device)

        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        if only_decoder:
            for p in self.encoder_decoder.module.encoder.parameters():
                p.requires_grad = False

        # mark "cover" as 1, "encoded" as 0
        # self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
        # self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

        # Mark "cover" as 0.9 (smooth label), "encoded" as 0.1
        self.label_cover = torch.full((batch_size, 1), 0.9, dtype=torch.float, device=device)
        self.label_encoded = torch.full((batch_size, 1), 0.0, dtype=torch.float, device=device)

        # optimizer (D learning rate can be slightly lower or equal)
        self.opt_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr
        )
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr*0.2)

        # learning rate schedulers (Cosine Annealing)
        self.scheduler_encoder_decoder = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_encoder_decoder, T_max=total_epochs, eta_min=1e-6
        )
        self.scheduler_discriminator = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_discriminator, T_max=total_epochs, eta_min=1e-6
        )

        # loss function
        self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
        self.criterion_MSE = nn.MSELoss().to(device)

        # weight of encoder-decoder loss
        self.discriminator_weight = 0.01
        self.encoder_weight = 1.5
        self.decoder_weight = 1.0

    def step_scheduler(self):
        """Call this once at the end of each training epoch."""
        self.scheduler_encoder_decoder.step()
        self.scheduler_discriminator.step()

    def get_last_lr(self):
        """Helper to log learning rates."""
        return {
            "lr_enc_dec": self.scheduler_encoder_decoder.get_last_lr()[0],
            "lr_disc": self.scheduler_discriminator.get_last_lr()[0]
        }

    def train(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.train()
        self.discriminator.train()

        with torch.enable_grad():
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            '''
            train discriminator
            '''
            self.opt_discriminator.zero_grad()

            d_label_cover = self.discriminator(images)
            d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
            d_cover_loss.backward()

            d_label_encoded = self.discriminator(encoded_images.detach())
            d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
            d_encoded_loss.backward()

            self.opt_discriminator.step()

            '''
            train encoder and decoder
            '''
            self.opt_encoder_decoder.zero_grad()

            g_label_decoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

            g_loss_on_encoder = self.criterion_MSE(encoded_images, images)
            g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

            g_loss = (self.discriminator_weight * g_loss_on_discriminator + 
                      self.encoder_weight * g_loss_on_encoder + 
                      self.decoder_weight * g_loss_on_decoder)

            g_loss.backward()
            self.opt_encoder_decoder.step()

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * ssim_loss(
                encoded_images.detach(),
                images,
                window_size=5,
                reduction="mean"
            )

        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": -psnr,
            "ssim": ssim,
            "g_loss": g_loss.item(),
            "g_loss_on_discriminator": g_loss_on_discriminator.item(),
            "g_loss_on_encoder": g_loss_on_encoder.item(),
            "g_loss_on_decoder": g_loss_on_decoder.item(),
            "d_cover_loss": d_cover_loss.item(),
            "d_encoded_loss": d_encoded_loss.item()
        }
        return result
    
    def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.train()

        with torch.enable_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            '''
            train encoder and decoder
            '''
            self.opt_encoder_decoder.zero_grad()

            # RESULT : the decoded message should be similar to the raw message
            g_loss = self.criterion_MSE(decoded_messages, messages)

            g_loss.backward()
            self.opt_encoder_decoder.step()

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * ssim_loss(
                encoded_images.detach(),
                images,
                window_size=5,
                reduction="mean"
            )

        '''
        decoded message error rate
        '''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": -psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": 0.,
            "g_loss_on_encoder": 0.,
            "g_loss_on_decoder": 0.,
            "d_cover_loss": 0.,
            "d_encoded_loss": 0.
        }
        return result

    def validation(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.eval()
        self.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            '''
            validate discriminator
            '''
            # RAW : target label for image should be "cover"(1)
            d_label_cover = self.discriminator(images)
            d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

            # GAN : target label for encoded image should be "encoded"(0)
            d_label_encoded = self.discriminator(encoded_images.detach())
            d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

            '''
            validate encoder and decoder
            '''

            # GAN : target label for encoded image should be "cover"(0)
            g_label_decoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

            # RESULT : the decoded message should be similar to the raw message
            g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

            # full loss
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
                     self.decoder_weight * g_loss_on_decoder

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * ssim_loss(
                encoded_images.detach(),
                images,
                window_size=5,
                reduction="mean"
            )

        '''
        decoded message error rate
        '''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": -psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder": g_loss_on_encoder,
            "g_loss_on_decoder": g_loss_on_decoder,
            "d_cover_loss": d_cover_loss,
            "d_encoded_loss": d_encoded_loss
        }

        return result, (images, encoded_images, noised_images, messages, decoded_messages)

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[0]

        message = message.gt(0.5)
        decoded_message = decoded_message.gt(0.5)
        error_rate = float(sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages):
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
        error_rate /= batch_size
        return error_rate

    def save_model(self, path_encoder_decoder: str, path_discriminator: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        checkpoint = torch.load(path_encoder_decoder, map_location=self.device)
        if hasattr(self.encoder_decoder, "module"):
            self.encoder_decoder.module.load_state_dict(checkpoint, strict=False)
        else:
            self.encoder_decoder.load_state_dict(checkpoint, strict=False)

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator))
