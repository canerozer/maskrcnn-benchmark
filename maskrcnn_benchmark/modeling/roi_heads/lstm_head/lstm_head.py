import torch
from torch import nn

from .lstm_box_predictors import make_lstm_predictor


class LSTMHead():
    """
    LSTM Head to be placed right at the end of Mask/Faster R-CNN.
    Firstly, the correct bounding box will be selected using the result
      of previous frame.
    Secondly, the following attributes will be propogated to an LSTM layer:
        - Features corresponding the bounding box
        - (x, y, w, h) of a bounding box
    Lastly, backward pass will be performed by selecting the most appropiate
bounding box after adding a Gaussian noise where necessary. Only LSTM layers
will be trained during the backward pass.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.predictor = make_lstm_predictor(cfg)
        self.loss_evaluator = make_lstm_loss_evaluator(cfg)

    def forward(self, rpn_features, coords, targets=None):
        features = torch.stack([rpn_features, coords], dim=2)
        confidence, coordinates = self.predictor(features)

        #if cfg.MODEL.LSTM_TRAIN_ONLY:
        #    with torch.no_grad():
def build_roi_mask_head(cfg):
    return LSTMHead(cfg)
