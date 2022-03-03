import torch
import numpy as np
import os
import sys
import math
import cv2



class SSIM():
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    # @staticmethod #bad
    # def __call__(self, img1, img2):
    #     if not img1.shape == img2.shape:
    #         raise ValueError("Input images must have the same dimensions.")
    #     if img1.ndim == 2:  # Grey or Y-channel image
    #         return self._ssim(img1, img2)
    #     elif img1.ndim == 3:
    #         if img1.shape[2] == 3:
    #             ssims = []
    #             for i in range(3):
    #                 ssims.append(self._ssim(img1, img2))
    #             return np.array(ssims).mean()
    #         elif img1.shape[2] == 1:
    #             return self._ssim(np.squeeze(img1), np.squeeze(img2))
    #     else:
    #         raise ValueError("Wrong input image dimensions.")


    @classmethod
    def __call__(cls, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return cls._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[0] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(cls._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[0] == 1:
                return cls._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        img1 = 255 * (img1.astype(float) - np.min(img1)) / float(np.max(img1) - np.min(img1))
        img2 = 255 * (img2.astype(float) - np.min(img2)) / float(np.max(img2) - np.min(img2))

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


def coo_scipy_to_pytorch(sci_P_coo):
    values = torch.FloatTensor(sci_P_coo.data)
    indices = torch.LongTensor(np.int64(np.vstack((sci_P_coo.row, sci_P_coo.col))))
    shape = sci_P_coo.shape
    P = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
    return P

def to_uint8(im1):
    im1 = 255 * (im1.astype(float) - np.min(im1)) / float(np.max(im1) - np.min(im1))
    # im1 = im1.astype(np.uint8)
    return im1

