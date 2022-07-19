import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
from bs4 import BeautifulSoup
from google.colab.patches import cv2_imshow


class textdetection:
    def __init__(self) -> None:
        self.net = CRAFT()
        self.cuda = False
        self.poly = False
        self.text_threshold = 0.7
        self.link_threshold = 0.4
        self.low_text = 0.4

        print('Loading weights from checkpoint (' + 'craft_mlt_25k.pth' + ')')
        if self.cuda==True:
            self.net.load_state_dict(self.__copyStateDict(torch.load('models/CRAFT-pytorch/craft_mlt_25k.pth')))
        else:
            self.net.load_state_dict(self.__copyStateDict(torch.load('models/CRAFT-pytorch/craft_mlt_25k.pth', map_location='cpu')))

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            ## cudnn.benchmark = False

        self.net.eval()
    
    @staticmethod
    def __copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def __test_net(self, image):
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        return boxes, polys, ret_score_text

    def predict(self, image):
      bboxes, _, _ = self.__test_net(image)
      return bboxes

textdetector = textdetection()