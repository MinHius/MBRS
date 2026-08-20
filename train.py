from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import *
from network.Network import *
from utils.load_train_setting import *

plt.switch_backend('Agg')

'''
setup plotting and logging directory
'''
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plot_log_dir = os.path.join("./run_log", timestamp)
os.makedirs(plot_log_dir, exist_ok=True)

history = {
    "epochs": [],
    "train": {
        "error_rate": [], "psnr": [], "ssim": [], "g_loss": [],
        "g_loss_on_discriminator": [], "g_loss_on_encoder": [],
        "g_loss_on_decoder": [], "d_cover_loss": [], "d_encoded_loss": []
    },
    "val": {
        "error_rate": [], "psnr": [], "ssim": [], "g_loss": [],
        "g_loss_on_discriminator": [], "g_loss_on_encoder": [],
        "g_loss_on_decoder": [], "d_cover_loss": [], "d_encoded_loss": []
    }
}

"""
loss_on_encoder: How good the encoder is at fooling the discriminator.
loss_on_decoder: How good the decoder is at recovering the message.
loss_on_discriminator: How good the discriminator is at distinguishing between cover and encoded images.

"""

def update_metric_plots(history_data, save_dir):
    epochs = history_data["epochs"]
    metrics = list(history_data["train"].keys())
    
    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(epochs, history_data["train"][metric], label="Train", color="tab:blue", marker="o", markersize=3)
        ax.plot(epochs, history_data["val"][metric], label="Val", color="tab:orange", marker="s", markersize=3)
        ax.set_title(metric.replace("_", " ").upper(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
    
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_curves.png"), dpi=200)
    plt.close(fig)

'''
train setup
'''
total_epochs = epoch_number + (train_continue_epoch if train_continue else 0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, total_epochs, with_diffusion, only_decoder)

train_dataset = MBRSDataset(os.path.join(dataset_path, "train"), H, W)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = MBRSDataset(os.path.join(dataset_path, "validation"), H, W)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

if train_continue:
    EC_path = f"results/{train_continue_path}/models/EC_{train_continue_epoch}.pth"
    D_path = f"results/{train_continue_path}/models/D_{train_continue_epoch}.pth"
    network.load_model(EC_path, D_path)

print(f"[*] Training initialized on: {device}", flush=True)
print(f"[*] Logs & Plots: {plot_log_dir}\n", flush=True)

total_train_steps = len(train_dataloader)

for raw_epoch in range(epoch_number):
    epoch = raw_epoch + (train_continue_epoch if train_continue else 0)

    running_result = {k: 0.0 for k in history["train"].keys()}
    train_start = time.time()

    # --- Training Loop ---
    for step, images in enumerate(train_dataloader):
        image = images.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        result = network.train(image, message) if not only_decoder else network.train_only_decoder(image, message)

        for key in result:
            v = result[key]
            running_result[key] += v.detach().item() if isinstance(v, torch.Tensor) else float(v)

        # Dynamic 1-line running train tracker
        current_step = step + 1
        loss_val = result.get('g_loss', 0.0)
        loss_val = loss_val.detach().item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
        err_val = result.get('error_rate', 0.0)
        err_val = err_val.detach().item() if isinstance(err_val, torch.Tensor) else float(err_val)

        sys.stdout.write(
            f"\r[Epoch {epoch:03d}/{total_epochs:03d}] "
            f"Train Step [{current_step:04d}/{total_train_steps:04d}] "
            f"| G-Loss: {loss_val:.4f} | Bit-Err: {err_val:.4f}"
        )
        sys.stdout.flush()

    train_elapsed = time.time() - train_start
    train_metrics = {k: running_result[k] / total_train_steps for k in running_result}
    
    for k, v in train_metrics.items():
        history["train"][k].append(v)

    # Overwrite the running progress line with the finalized train summary
    sys.stdout.write(
        f"\r[Epoch {epoch:03d}/{total_epochs:03d}] (TRAIN {train_elapsed:.1f}s) "
        f"G-Loss: {train_metrics['g_loss']:.4f} | Err: {train_metrics['error_rate']:.4f} | "
        f"PSNR: {train_metrics['psnr']:.2f} | SSIM: {train_metrics['ssim']:.4f}\n"
    )
    sys.stdout.flush()

    # Write file log
    with open(f"{result_folder}/train_log.txt", "a") as f:
        f.write(f"Epoch {epoch} : {int(train_elapsed)}s\n" + ",".join(f"{k}={v}" for k, v in train_metrics.items()) + "\n")

    # --- Validation Loop ---
    val_result = {k: 0.0 for k in history["val"].keys()}
    val_start = time.time()

    saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False)
    saved_all = None

    for i, images in enumerate(val_dataloader):
        image = images.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message)

        for key in result:
            v = result[key]
            val_result[key] += v.detach().item() if isinstance(v, torch.Tensor) else float(v)

        if i in saved_iterations:
            if saved_all is None:
                saved_all = get_random_images(image, encoded_images, noised_images)
            else:
                saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

    save_images(saved_all, epoch, f"{result_folder}images/", resize_to=(W, H))

    val_elapsed = time.time() - val_start
    total_val_steps = len(val_dataloader)
    val_metrics = {k: val_result[k] / total_val_steps for k in val_result}

    for k, v in val_metrics.items():
        history["val"][k].append(v)
    history["epochs"].append(epoch)

    # Print 1-line validation summary
    sys.stdout.write(
        f"                   (VAL   {val_elapsed:.1f}s) "
        f"G-Loss: {val_metrics['g_loss']:.4f} | Err: {val_metrics['error_rate']:.4f} | "
        f"PSNR: {val_metrics['psnr']:.2f} | SSIM: {val_metrics['ssim']:.4f}\n"
    )
    sys.stdout.flush()

    with open(f"{result_folder}/val_log.txt", "a") as f:
        f.write(f"Epoch {epoch} : {int(val_elapsed)}s\n" + ",".join(f"{k}={v}" for k, v in val_metrics.items()) + "\n")

    # --- Step Learning Rate Scheduler ---
    network.step_scheduler()
    lrs = network.get_last_lr()
    sys.stdout.write(f"[*] Updated LR -> G: {lrs['lr_enc_dec']:.2e} | D: {lrs['lr_disc']:.2e}\n")
    sys.stdout.flush()

    # --- Save Model & Update Curves ---
    path_model = f"{result_folder}models/"
    network.save_model(f"{path_model}EC_{epoch}.pth", f"{path_model}D_{epoch}.pth")

    if (raw_epoch + 1) % 2 == 0 or (raw_epoch + 1) == epoch_number:
        update_metric_plots(history, plot_log_dir)

    # Clean epoch boundary spacing
    print("", flush=True)