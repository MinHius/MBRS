from torch.utils.data import DataLoader
from utils import *
from network.Network import *

from utils.load_test_setting import *

'''
test
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion)
EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
network.load_model_ed(EC_path)

test_dataset = MBRSDataset(os.path.join(dataset_path, "test"), H, W)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print("\nStart Testing : \n\n")

test_result = {
	"error_rate": 0.0,
	"psnr": 0.0,
	"ssim": 0.0
}

start_time = time.time()

saved_iterations = np.random.choice(np.arange(len(test_dataset)), size=save_images_number, replace=False)
saved_all = None

num = 0
for i, (images, images_hr) in enumerate(test_dataloader):
    image = images.to(device)
    image_hr = images_hr.to(device)
    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

    '''
    test
    '''
    network.encoder_decoder.eval()
    network.discriminator.eval()


    with torch.no_grad():
        images = images.to(network.device)       # 128x128
        messages = message.to(network.device)
        images_hr = images_hr.to(network.device) # original HR

        # 1. Encode at native MBRS resolution
        encoded_lr = network.encoder_decoder.module.encoder(
            images,
            messages
        )

        # 2. Extract watermark residual
        residual_lr = encoded_lr - images

        # 3. Transport residual to HR
        residual_hr = F.interpolate(
            residual_lr,
            size=images_hr.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # 4. Composite onto original HR image
        watermarked_hr = images_hr + residual_hr * strength_factor

        # 5. Decode
        noised_images = network.encoder_decoder.module.noise(
            [watermarked_hr, images]
        )

        if isinstance(noised_images, (tuple, list)):
            noised_images = noised_images[0]

        # 6. Bring HR composite back to MBRS resolution
        noised_images = F.interpolate(
            noised_images,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        decoded_messages = network.encoder_decoder.module.decoder(
            noised_images
        )

        # 7. Image quality should be measured at HR
        psnr = kornia.losses.psnr_loss(
            watermarked_hr.detach(),
            images_hr,
            2
        ).item()

        ssim = 1 - 2 * ssim_loss(
            watermarked_hr.detach(),
            images_hr,
            window_size=5,
            reduction="mean"
        )

    '''
    decoded message error rate
    '''
    error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)

    result = {
        "error_rate": error_rate,
        "psnr": -psnr,
        "ssim": ssim,
    }

    for key in result:
        test_result[key] += float(result[key])

    num += 1

    # if i in saved_iterations:
    #     if saved_all is None:
    #         saved_all = get_random_images(image, encoded_images, noised_images)
    #     else:
    #         saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

    '''
    test results
    '''
    content = "Image " + str(i) + " : \n"
    for key in test_result:
        content += key + "=" + str(result[key]) + ","
    content += "\n"

    with open(test_log, "a") as file:
        file.write(content)

    print(content)

'''
test results
'''
content = "Average : \n"
for key in test_result:
	content += key + "=" + str(test_result[key] / num) + ","
content += "\n"

with open(test_log, "a") as file:
	file.write(content)

print(content)
# save_images(saved_all, "test", result_folder + "images/", resize_to=(W, H))
