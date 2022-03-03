import numpy as np
import torch
import scipy.io
# import visdom
import pydicom
import cv2
import matplotlib.pyplot as plt
import os
import sys
import time

from torch import nn
# from bm3d import bm3d
from tensorboardX import SummaryWriter

from net import UNet, pure_net, DnCNN
from utils import coo_scipy_to_pytorch
from config import parser


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def add_noise(model, param_noise_sigma, learning_rate, device):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        noise = torch.randn(n.size()) * param_noise_sigma * learning_rate
        noise = noise.to(device).float()
        n.data = n.data + noise

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2021)

def main(args):
    try:
        gt_path = '....dcm'
        sino_path = '....npy'


        dir_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        dir = dir_time + args.task + '_gen=unet_' + args.id + '_p={}'.format(args.p_sample)
        if not os.path.exists(os.path.join('./', dir)):
            os.mkdir(os.path.join('./', dir))

        # load sino and gt
        sino = np.load(sino_path)
        sino = torch.from_numpy(sino).to(args.device).float()

        gt = pydicom.read_file(gt_path).pixel_array.astype(np.float32)
        gt = cv2.resize(gt, (args.img_size, args.img_size))

        # load sys matrix
        lmat = scipy.io.loadmat('place system_matrix here')
        S = lmat['place the name of system matrix here']  
        S_coo = S.tocoo() 
        P = coo_scipy_to_pytorch(S_coo)
        P = P.to(args.device)


        # init


        z = torch.matmul(P.t(), sino)

        # net
        net = UNet(1, 1)
        net.to(args.device)

        # denoiser
        model = DnCNN(1, 17)
        model.to(args.device)
        # model = nn.DataParallel(model).to(device)
        # model.load_state_dict(torch.load('./net.pth'))
        # model.eval()

        # optimizer
        criterion = torch.nn.MSELoss()
        optimizer1 = torch.optim.Adam(net.parameters(), lr=args.lr_G, weight_decay = args.weight_decay)
        optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr_D)

        # log
        log = SummaryWriter()

        # visdom
        # viz = visdom.Visdom(env='DIP_PET')
        # image_win = viz.image(np.random.rand(1000, 1000),
        #                       opts=dict(caption='First random', store_history=True, title='Pick your random!'))

        # record list
        loss_list = np.zeros(args.epochs)
        psnr_list = np.zeros(args.epochs)
        max_psnr = 0.0

        # train
        for i in range(args.epochs):
            x = net(z.reshape(1, 1, args.img_size, args.img_size))
            y_hat = torch.matmul(P * args.sp_p_sample, x.reshape(args.img_size * args.img_size, 1))
            penalty = args.my_lambda * torch.abs(torch.sum(x * (x - model(x))))
            loss = criterion(y_hat, sino) + penalty
            #         loss = criterion(y_hat, sino)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            print(loss.item())
            optimizer1.step()
            optimizer2.step()

            add_noise(net, args.param_noise_sigma, args.learning_rate, args.device)

            # viz.image(to_uint8(x.squeeze().detach().cpu().numpy()), win=image_win)

            # eval
            mse = torch.nn.MSELoss()(x.squeeze(), torch.from_numpy(gt).to(args.device))
            data_range = np.max(np.max(gt, 1)).astype(np.float32)
            psnr = 10 * torch.log10(data_range ** 2 / mse)
            print('epoch: ', i, 'psnr: ', psnr)
            if psnr > max_psnr:
                max_psnr = psnr
                np.save(os.path.join(dir, 'BESTimage_' + dir + '.npy'), x.detach().cpu().numpy().squeeze())

            # plot
            log.add_scalars(dir, {'psnr': psnr,
                                  'loss': loss}, i)

            # save record list
            loss_list[i] = loss
            psnr_list[i] = psnr

            if i % 100 == 0:
                np.save(dir + '/epoch{}.npy'.format(i), x.squeeze().detach().cpu().numpy())
    except KeyboardInterrupt:
        # save the net
        torch.save(net.state_dict(), os.path.join(dir, dir + '.pth'),
                   _use_new_zipfile_serialization=False)

        np.save(os.path.join(dir, 'loss_' + dir + '.npy'), loss_list)
        np.save(os.path.join(dir, 'psnr_' + dir + '.npy'), psnr_list)
        np.save(os.path.join(dir, 'Interruptimage_' + dir + '.npy'), x.detach().cpu().numpy().squeeze())

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    # save the net
    torch.save(net.state_dict(), os.path.join(dir, dir + '.pth'),
               _use_new_zipfile_serialization=False)

    np.save(os.path.join(dir, 'loss_' + dir + '.npy'), loss_list)
    np.save(os.path.join(dir, 'psnr_' + dir + '.npy'), psnr_list)
    np.save(os.path.join(dir, 'Finalimage_' + dir + '.npy'), x.detach().cpu().numpy().squeeze())

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)








