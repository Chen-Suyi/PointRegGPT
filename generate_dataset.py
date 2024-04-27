from denoising_diffusion_pytorch.successive_ddnm_diffusion import Unet, GaussianDiffusion, Generator
from depth_correction_pytorch.depth_correction import MaskUnet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='checkpoint to load',
                    required=True)
parser.add_argument('--dataset_name',
                    default='generated_dataset',
                    type=str,
                    help='')
parser.add_argument('--start_scene_index',
                    '-start',
                    default=0,
                    type=int,
                    help='scenes index to start')
parser.add_argument('--stop_scene_index',
                    '-stop',
                    default=1,
                    type=int,
                    help='scenes index to stop')
parser.add_argument('--num_samples',
                    default=1,
                    type=int,
                    help='sample numbers for each scene')
args = parser.parse_args()

model = Unet(dim=64, param_cond_dim=4, dim_mults=(1, 2, 4, 8), channels=1)

diffusion = GaussianDiffusion(
    model,
    image_size=256,
    timesteps=1000,  # number of steps
    sampling_timesteps=
    250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type='l1',  # L1 or L2
    objective='pred_x0',
    beta_schedule='sigmoid',
    ddim_sampling_eta=1.0,
    is_ddnm_sampling=True)

generator = Generator(
    diffusion,
    '/data/data/3DMatch-RGBD/train', # path to 3DMatch RGB-D training data
    batch_size=4,
    ema_decay=0.995,  # exponential moving average decay
    results_folder='./successive_ddnm_diffusion_results',
    samples_folder='./{}/data'.format(
        args.dataset_name),  #'./generated_dataset/data'
    amp=False  # turn on mixed precision
)

generator.load("{}".format(args.resume))
depth_correction = MaskUnet(dim=64, dim_mults=(1, 2, 4, 8))
generator.generate(start_scene_index=args.start_scene_index,
                    stop_scene_index=args.stop_scene_index,
                    num_samples=args.num_samples,
                    has_refine_step=False,
                    depth_correction=depth_correction)