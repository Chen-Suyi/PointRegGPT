from denoising_diffusion_pytorch.successive_ddnm_diffusion import Unet, GaussianDiffusion, Tester

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume',
                    default=None,
                    type=int,
                    help='checkpoint to load')
parser.add_argument('--num_scenes',
                    default=4,
                    type=int,
                    help='how many scenes to generate')
parser.add_argument('--num_samples',
                    default=4,
                    type=int,
                    help='sample numbers for each scene')
args = parser.parse_args()

model = Unet(dim=64, param_cond_dim=4, dim_mults=(1, 2, 4, 8), channels=1)

diffusion = GaussianDiffusion(
    model,
    image_size=256,
    timesteps=1000,  # number of steps
    sampling_timesteps=
    32,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type='l1',  # L1 or L2
    objective='pred_x0',
    beta_schedule='sigmoid',
    ddim_sampling_eta=1.0,
    is_ddnm_sampling=True)

tester = Tester(
    diffusion,
    batch_size=4,
    ema_decay=0.995,  # exponential moving average decay
    results_folder='./successive_ddnm_diffusion_results',
    samples_folder='./successive_ddnm_diffusion_samples',
    amp=False  # turn on mixed precision
)

tester.load("{}".format(args.resume))
tester.sample(num_scenes=args.num_scenes, num_samples=args.num_samples)
