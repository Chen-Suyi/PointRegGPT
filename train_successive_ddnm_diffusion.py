from denoising_diffusion_pytorch.successive_ddnm_diffusion import Unet, GaussianDiffusion, Trainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume',
                    default=None,
                    type=int,
                    help='checkpoint to load')
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
    ddim_sampling_eta=0.,
    is_ddnm_sampling=True)

trainer = Trainer(
    diffusion,
    '/path/to/3DMatch-RGBD/train', # path to 3DMatch RGB-D training data
    train_batch_size=32, # required 48GB CUDA memory
    train_lr=8e-5,
    train_num_steps=2000000,  # total training steps 2000000
    gradient_accumulate_every=2,  # gradient accumulation steps
    augment_horizontal_flip=True,
    ema_decay=0.995,  # exponential moving average decay
    save_and_sample_every=1000,
    num_samples=25,
    results_folder='./successive_ddnm_diffusion_results',
    samples_folder='./successive_ddnm_diffusion_samples',
    amp=False,  # turn on mixed precision
    calculate_fid=False  # whether to calculate fid during training
)

if args.resume is None:
    pass
else:
    trainer.load("{}".format(args.resume))
trainer.train()