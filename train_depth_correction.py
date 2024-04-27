from depth_correction_pytorch.depth_correction import MaskUnet, MaskTrainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='checkpoint to load')
args = parser.parse_args()

model = MaskUnet(dim=64, dim_mults=(1, 2, 4, 8))

trainer = MaskTrainer(
    model,
    './dataset/depth_correction', # path to depth correction dataset
    image_size=256,
    train_batch_size=4,  # for 12GB CUDA memory
    train_lr=4e-5,
    lr_gamma=0.95,
    epochs=100,
    results_folder='./depth_correction_results',
    samples_folder='./depth_correction_samples')

if args.resume is None:
    pass
else:
    trainer.load("{}".format(args.resume))
trainer.train_and_eval()
# trainer.test()