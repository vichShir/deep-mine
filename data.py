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


class Encoder():

    def __init__(self, df_feedbacks, df_products, df_users, description_vocab, review_vocab):
        self.df_feedbacks = df_feedbacks
        self.df_products = df_products
        self.df_users = df_users
        self.description_vocab = description_vocab
        self.review_vocab = review_vocab

    def process(self, batch):
        u_ls = []
        pos_ids = []
        pos_imgs = []
        pos_descs = []
        pos_revs = []
        neg_ids = []
        neg_imgs = []
        neg_descs = []
        neg_revs = []
        for data in batch:
            u = data[0]
            i = data[1]
            j = data[2]

            user_key = self.df_users[self.df_users['index'] == u]['reviewerID'].iloc[0]
            pos_feedback = self.df_feedbacks[self.df_feedbacks['reviewerID'] == user_key].iloc[0]

            image_name = os.path.basename(pos_feedback['imageURL'])
            p_image = load_image(os.path.join('./data/images/', image_name))
            p_description = self.description_vocab.transform(pos_feedback['description']).squeeze(0).float()
            p_review = self.review_vocab.transform(pos_feedback['reviewText']).squeeze(0).float()

            imgs = []
            descs = []
            revs = []
            for neg_item_id in j:
                item_key = self.df_products[self.df_products['index'] == neg_item_id]['asin'].iloc[0]

                try:
                    neg_feedback = self.df_feedbacks[self.df_feedbacks['asin'] == item_key].sample(1).iloc[0]

                    image_name = os.path.basename(neg_feedback['imageURL'])
                    n_image = load_image(os.path.join('./data/images/', image_name))
                    n_description = self.description_vocab.transform(neg_feedback['description']).squeeze(0).float()
                    n_review = self.review_vocab.transform(neg_feedback['reviewText']).squeeze(0).float()
                except:
                    n_image = torch.zeros(3, 40, 40)
                    n_description = torch.zeros((12916,))
                    n_review = torch.zeros((29678,))

                imgs.append(n_image.unsqueeze(0))
                descs.append(n_description.unsqueeze(0))
                revs.append(n_review.unsqueeze(0))

            u_ls.append(torch.tensor(u).unsqueeze(0))
            pos_ids.append(torch.tensor(i).unsqueeze(0))
            pos_imgs.append(p_image.unsqueeze(0))
            pos_descs.append(p_description.unsqueeze(0))
            pos_revs.append(p_review.unsqueeze(0))
            neg_ids.append(torch.tensor(j))
            neg_imgs.append(torch.cat(imgs))
            neg_descs.append(torch.cat(descs))
            neg_revs.append(torch.cat(revs))

        return {
            'user_ids': torch.cat(u_ls),
            'pos_item': {
                'ids': torch.cat(pos_ids),
                'images': torch.cat(pos_imgs),
                'descriptions': torch.cat(pos_descs),
                'reviews': torch.cat(pos_revs)
            },
            'neg_item': {
                'ids': torch.cat(neg_ids, dim=0),
                'images': torch.cat(neg_imgs),
                'descriptions': torch.cat(neg_descs),
                'reviews': torch.cat(neg_revs)
            }
        }


def load_image(path, resize=(40, 40)):
    numpy_img = np.array(Image.open(path).resize(resize))
    if numpy_img.ndim == 2:
        numpy_img = np.repeat(numpy_img[:, :, np.newaxis], 3, axis=2)
    return torch.from_numpy(numpy_img).view(3, 40, 40) / 255


class DeepMINEDataset(Dataset):

    def __init__(self, df_feedbacks, df_products, df_users, n_neg_items=1):
        self.df_feedbacks = df_feedbacks
        self.df_products = df_products
        self.df_users = df_users
        self.n_neg_items = n_neg_items

    def __len__(self):
        return self.df_feedbacks.shape[0]

    def __getitem__(self, idx):
        try:
            pos_feedback = self.df_feedbacks.iloc[idx]
            u = self.df_users[self.df_users['reviewerID'] == pos_feedback['reviewerID']]['index'].iloc[0]
            i = self.df_products[self.df_products['asin'] == pos_feedback['asin']]['index'].iloc[0]

            user_key = pos_feedback['reviewerID']
            user_products = list(self.df_feedbacks.query("reviewerID == @user_key")['asin'].unique())  # list all products user has a feedback
            found_neg_item = False
            # while found_neg_item is False:
            n_item_key = self.df_products.query("asin != @user_products").sample(self.n_neg_items)['asin'].to_list()  # get a random negative product
            all_neg_feedbacks = self.df_feedbacks[self.df_feedbacks['asin'].isin(n_item_key)]
                # if all_neg_feedbacks.shape[0] > 0 and all_neg_feedbacks['asin'].nunique() == self.n_neg_items:
                #     found_neg_item = True
            j = self.df_products[self.df_products['asin'].isin(n_item_key)]['index'].to_list()
        except Exception as e:
            print('FAILED data loader:', e)
            import ipdb; ipdb.set_trace()

        return u, i, j