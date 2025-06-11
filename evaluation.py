import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score

import os
import time
import json
import random
from PIL import Image
from tqdm import tqdm


def hit_score(item_true, items_pred):
    if item_true in items_pred:
        return 1
    return 0


def evaluate(model, dataloader, device, topk=1):
    HR = []
    AUC = []
    model.eval()
    for idx, data in enumerate(pbar := tqdm(dataloader)):
        # user_id, positive_item, negative_item = data
        user_id = data['user_ids']
        positive_item = data['pos_item']
        negative_item = data['neg_item']

        p_item, p_image, p_description, p_review = positive_item['ids'], positive_item['images'], positive_item['descriptions'], positive_item['reviews']
        n_item, n_image, n_description, n_review = negative_item['ids'], negative_item['images'], negative_item['descriptions'], negative_item['reviews']

        with torch.no_grad():
            pred_p = model.recommend(user_id.to(device), p_item.to(device), p_image.to(device), p_description.to(device), p_review.to(device))
            pred_n = model.recommend(user_id.repeat(n_item.shape[0]).to(device), n_item.to(device), n_image.to(device), n_description.to(device), n_review.to(device))

        predictions = torch.cat([pred_p, pred_n], dim=0).squeeze().cpu()

        # hit-ratio
        _, indices = torch.topk(predictions, k=topk)
        recommends = torch.take(torch.cat([p_item, n_item], dim=0).squeeze(), indices)
        hit = hit_score(p_item, recommends)

        # roc-auc
        roc_auc = roc_auc_score(np.array([1] + [0]*(predictions.shape[0]-1)), predictions.numpy())

        HR.append(hit)
        AUC.append(roc_auc)

        pbar.set_postfix(hit=np.mean(HR), roc_auc=np.mean(AUC))
        pbar.update()

        if idx >= 5000:
            break
    print('Hit-Ratio:', np.mean(HR))
    print('ROC-AUC:', np.mean(AUC))