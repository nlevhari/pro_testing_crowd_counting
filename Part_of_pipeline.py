from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model_for_pipeline
import os
import warnings
warnings.filterwarnings('ignore')


def build_device_model_transform(gpu_id='', device_type='cpu', weight_path='./weights/SHTechA.pth'):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpu_id)

    device = torch.device(device_type)
    # get the P2PNet
    model = build_model_for_pipeline()
    # move to GPU
    model.to(device)
    # load trained model
    if weight_path is not None:
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return device, model, transform


def predict_image(im, device, model, transform):
    img_path = im
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # pre-processing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    #
    # outputs_points = outputs['pred_points'][0]
    return points, predict_cnt
