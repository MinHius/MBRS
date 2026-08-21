from torch.utils.data import DataLoader
from utils import *
from network.Network import *
from kornia.losses import ssim_loss

from utils.load_test_setting import *

'''
test
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network(H, W, message_length, noise_layers, device, batch_size, lr, 100, with_diffusion)
EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
network.load_model_ed(EC_path)

test_dataset = MBRSDataset(os.path.join(dataset_path, "test"), H, W)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print("\nStart Testing : \n\n")

test_result = {
	"max_ber": 0.0,
	"mean_ber": 0.0,
	"min_ber": 0.0,
	"max_psnr": 0.0,
	"mean_psnr": 0.0,
	"min_psnr": 0.0,
	"max_ssim": 0.0,
	"mean_ssim": 0.0,
	"min_ssim": 0.0,
}

start_time = time.time()

saved_iterations = np.random.choice(np.arange(len(test_dataset)), size=save_images_number, replace=False)
saved_all = None

num = 0
test_log = "./results/MBRS_m30_Combined([Identity(),Jpeg(50),Crop(0.7,0.7),Cropout(0.2,0.2),Dropout(0.2),GN(0.0,0.03)])__2026_08_18__10_57_25/resize.txt"
for i, images in enumerate(test_dataloader):
	image = images.to(device)
	message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

	'''
	test
	'''
	network.encoder_decoder.eval()
	network.discriminator.eval()
	min_ber = 1.0
	min_psnr = 90.0
	min_ssim = 1.0
	max_ber = 0.0
	max_psnr = 0.0
	max_ssim = 0.0

	with torch.no_grad():
		# use device to compute
		images, messages = images.to(network.device), message.to(network.device)

		encoded_images = network.encoder_decoder.module.encoder(images, messages)
		encoded_images = images + (encoded_images - image) * strength_factor
		print("Before noise:", encoded_images.min().item(), encoded_images.max().item())
		noised_images = network.encoder_decoder.module.noise([encoded_images, images])
		print("After noise: ", noised_images.min().item(), noised_images.max().item())

		decoded_messages = network.encoder_decoder.module.decoder(noised_images)

		# psnr
		psnr = -(kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item())

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
	error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)

	result = {
		"max_ber": max(max_ber, float(error_rate)),
		"mean_ber": float(error_rate),
		"min_ber": min(min_ber, float(error_rate)),
		"max_psnr": max(max_psnr, float(psnr)),
		"mean_psnr": float(psnr),
		"min_psnr": min(min_psnr, float(psnr)),
		"max_ssim": max(max_ssim, float(ssim)),
		"mean_ssim": float(ssim),
		"min_ssim": min(min_ssim, float(ssim)),
	}

	for key in result:
		print(f"{key}: {result[key]}")
		test_result[key] += result[key]

	num += 1

	if i in saved_iterations:
		if saved_all is None:
			saved_all = get_random_images(image, encoded_images, noised_images)
		else:
			saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

	'''
	test results
	'''
	content = "Image " + str(i) + " : \n"
	for key in test_result:
		content += key + "=" + str(result[key]) + ","
	content += "\n"

	# with open(test_log, "a") as file:
	# 	file.write(content)

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
save_images(saved_all, "test", result_folder + "images/", resize_to=(W, H))
