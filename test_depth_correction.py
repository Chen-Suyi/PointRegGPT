from depth_correction_pytorch.depth_correction import MaskUnet, MaskTester

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='checkpoint to load')
args = parser.parse_args()

model = MaskUnet(dim=64, dim_mults=(1, 2, 4, 8))

tester = MaskTester(model,
                    '/path/to/3DMatch-RGBD/test', # path to 3DMatch RGB-D test set
                    image_size=256,
                    results_folder='./depth_correction_results',
                    samples_folder='./depth_correction_samples')

if args.resume is None:
    pass
else:
    tester.load("{}".format(args.resume))
tester.test()