# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix
from pycm import *
import matplotlib.pyplot as plt
import numpy as np




def misc_measures(confusion_matrix):
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []

    for i in range(1, confusion_matrix.shape[0]):
        cm1 = confusion_matrix[i]
        
        # Accuracy
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        
        # Sensitivity
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1]) if (cm1[1,0]+cm1[1,1]) > 0 else 0
        sensitivity.append(sensitivity_)
        
        # Specificity
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0]) if (cm1[0,1]+cm1[0,0]) > 0 else 0
        specificity.append(specificity_)
        
        # Precision
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1]) if (cm1[1,1]+cm1[0,1]) > 0 else 0
        precision.append(precision_)
        
        # G-Mean
        G.append(np.sqrt(sensitivity_ * specificity_) if sensitivity_ > 0 and specificity_ > 0 else 0)
        
        # F1-Score
        F1_score_2.append(2 * precision_ * sensitivity_ / (precision_ + sensitivity_) if (precision_ + sensitivity_) > 0 else 0)
        
        # MCC
        denominator = np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0]) / denominator if denominator > 0 else 0
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_
