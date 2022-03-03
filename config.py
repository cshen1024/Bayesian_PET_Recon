import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda', help='device to use')
parser.add_argument("--epochs", type=int, default=12000, help="number of epochs of training")
parser.add_argument('--my_lambda', type=float, default='0.01', help='trade off between fidelity and penalty term')
parser.add_argument('--lr_G', type=float, default=1e-3, help='learning rate of Generator')
parser.add_argument('--lr_D', type=float, default=1e-3, help='learning rate of Denoiser')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate in SGLD')
parser.add_argument('--param_noise_sigma', type=float, default=1e-2, help='S.D. of noise in SGLD')
parser.add_argument('--p_sample', type=float, default=1, help='count level')
parser.add_argument('--sp_p_sample', type=float, default=1, help='adjusted count level')
parser.add_argument('--id', type=str, default='6-099', help='image id to be reconstructed')
parser.add_argument('--task', type=str, default='DeepRED_SGLD', help='task mode')
parser.add_argument('--weight_decay', type=float, default='5e-8', help='weight_decay')
parser.add_argument('--img_size', type=int, default='128', help='img_size')











