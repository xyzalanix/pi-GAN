import argparse
import math
import glob
import numpy as np
import imageio.v2 as imageio
import cv2
import numpy as np
import skimage
import sys
import os
import random
import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import time
from datetime import datetime
from tqdm import tqdm

import curriculums

persp = T.RandomPerspective(distortion_scale=0.5, p=0.5, fill=-1)
blur = T.GaussianBlur(kernel_size=31, sigma=10)
inv = T.RandomInvert(p=0.4)
rot = T.RandomRotation(degrees=(0, 180), fill=-1)
jitter = T.ColorJitter(brightness=10, contrast=10, saturation=0, hue=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    # plt.imshow(tensor_img)
    # plt.show()

def generate_img(gen, z, **kwargs):

    with torch.no_grad():
        img, depth_map = generator.staged_forward(z, **kwargs)
        tensor_img = img.detach()

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        img = img.squeeze() * 0.7 + 0.7
    return img, tensor_img, depth_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--psi', type=float, default=0.8)
    parser.add_argument('--max_batch_size', type=int, default=58000000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = opt.psi
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    model = os.path.split(opt.path)
    model = os.path.splitext(model[1])
    model = model[0]

    os.makedirs(opt.output_dir, exist_ok=True)

    generator = torch.load(opt.path, map_location=torch.device(device))
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    face_angles = [1, 0, 2.5, 0, 5, 0] ## grid set
    face_angles_v = [1,-1,1,-1,1,-1] ## fleur config - [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # face_psi = [-0.8, 1., -0.8, 1., 0.8, -1.0, 0.8, 1.0, -0.8]
    face_seed = [11,12,13,14,15,16] #[43, 44, 997, 9, 96, 1, 2, 11, 12]
    fov = curriculum['fov'] + 46
    face_angles = [a + curriculum['h_mean'] for a in face_angles]
    face_angles_v = [a for a in face_angles_v]
    # face_psi = [a + curriculum['psi'] for a in face_psi]
    face_seed = [a for a in face_seed]

    for seed in tqdm(opt.seeds):
        images = []
        for i, (yaw, pitch, seed) in enumerate(zip(face_angles, face_angles_v, face_seed)):
            curriculum['h_mean'] = yaw
            curriculum['v_mean'] = pitch
            face_angles = random.sample(face_angles, len(face_angles))
            face_angles_v = random.sample(face_angles_v, len(face_angles_v))
            face_seed = random.sample(face_seed, len(face_seed))

            torch.manual_seed(seed)
            curriculum['fov'] = fov
            z = torch.randn((1, 256), device=device)
            img, tensor_img, depth_map = generate_img(generator, z, **curriculum)
            print(' •°*”˜ѕєє∂',seed,'Tile', i,'-', 'yaw:', yaw,'pitch:', pitch)
            # tensor_img = persp(tensor_img)
            # tensor_img = blur(tensor_img)
            # tensor_img = inv(tensor_img)
            # tensor_img = rot(tensor_img)
            images.append(tensor_img)
        output = os.path.join(opt.output_dir, f'{model}_seed{face_seed}_{opt.psi}_{opt.image_size}px')
        save_image(torch.cat(images), f'{output}.png', normalize=True, padding=0, pad_value=1.0, nrow=3)

        # https://sparrow.dev/numpy-pad/
        result = imageio.imread(f'{output}.png')
        height, width, channels = result.shape
        reflection = np.pad(result, ((0, 0), (width, 0), (0, 0)), mode='reflect')
        print('Reflecting Grid 1/2')
        skimage.io.imsave(f'{output}_REFLECTED.png', reflection)
        print('Cleanup...')
        os.remove(f'{output}.png')

        one = imageio.imread(f'{output}_REFLECTED.png')
        height, width, channels = one.shape
        double_reflection = np.pad(one, ((height, 0), (0, 0), (0, 0)), mode='reflect')
        print('Reflecting Grid 2/2')
        final_out = f'{output}_REFLECTED_AGAIN.png'
        skimage.io.imsave(final_out, double_reflection)
        print('Cleanup...')
        os.remove(f'{output}_REFLECTED.png')
        print('Done.')

        ## Feather edges attempt
        og = cv2.imread(final_out)
        og = cv2.cvtColor(og, cv2.COLOR_BGR2RGB)

        blurred_img = cv2.GaussianBlur(255-og, (71, 71), 0)
        mask = np.zeros(og.shape, np.uint8)

        gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 0, 120)
        blurred_edged = cv2.GaussianBlur(edged, (31, 31), 0)

        cv2.waitKey(0)
        # contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(blurred_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(mask, contours, -1, (255,255,255),4)
        cv2.drawContours(mask, contours, -1, (255,255,255),2)

        # mask = cv2.GaussianBlur(mask, (3, 3), 0)

        masked = np.where(mask==np.array([255, 255, 255]),blurred_img, og)
        date = datetime.now().strftime("%I%M%S")
        skimage.io.imsave(f'{output}_masked_{date}.png', masked)
        # skimage.io.imsave(f'{output}_mask.png', mask)
        os.remove(f'{output}_REFLECTED_AGAIN.png')
