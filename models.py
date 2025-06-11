# https://discuss.pytorch.org/t/how-to-share-weights-between-two-layers/55541/2
# https://www.kaggle.com/code/ignazio/autoencoder-for-text-in-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


class ImageAutoEncoder(nn.Module):

    def __init__(self, img_size=40, n_node=100, kernel_size=3, tie_weights=True):
        super().__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.fc3 = nn.Linear(25600, n_node, bias=True)

        # decoder
        self.fc4 = nn.Linear(n_node, 25600, bias=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=kernel_size, stride=1, padding=1)

        # utils
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU()  # TODO: change to sigmoid?
        # self.relu = nn.Sigmoid()

        # share encoder decoder weight matrices
        if tie_weights:
            self._tie_weights()

    def _tie_weights(self):
        self.fc4.weight.data = self.fc3.weight.data.transpose(0,1)
        self.conv5.weight.data = self.conv2.weight.data.transpose(0,1)
        self.conv6.weight.data = self.conv1.weight.data.transpose(0,1)

    def forward(self, x):
        # encoder
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.pool(h)
        h = self.fc3(h.reshape(-1, 25600))

        # decoder
        h = self.fc4(h).T
        h = h.reshape(-1, 64, 20, 20)
        h = self.upsample(h)
        h = self.relu(self.conv5(h))
        x_hat = self.relu(self.conv6(h))
        return x_hat

    def encode(self, x):
        # encoder
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.pool(h)
        h = self.fc3(h.reshape(-1, 25600))
        return h

    def decode(self, h):
        # decoder
        h = self.fc4(h).T
        h = h.reshape(-1, 64, 20, 20)
        h = self.upsample(h)
        h = self.relu(self.conv5(h))
        x_hat = self.relu(self.conv6(h))
        return x_hat


class TextAutoEncoder(nn.Module):

    def __init__(self, vocab_size, n_node=100, hidden_size=400, tie_weights=True):
        super().__init__()

        # encoder
        self.fc1 = nn.Linear(vocab_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, n_node, bias=True)

        # decoder
        self.fc3 = nn.Linear(n_node, hidden_size, bias=True)
        self.fc4 = nn.Linear(hidden_size, vocab_size, bias=True)

        # utils
        self.relu = nn.ReLU()  # TODO: change to sigmoid?
        # self.relu = nn.Sigmoid()

        # share encoder decoder weight matrices
        if tie_weights:
            self._tie_weights()

    def _tie_weights(self):
        self.fc3.weight.data = self.fc2.weight.data.transpose(0,1)
        self.fc4.weight.data = self.fc1.weight.data.transpose(0,1)

    def forward(self, x):
        # encoder
        h = self.relu(self.fc1(x))
        h = self.fc2(h)

        # decoder
        h = self.relu(self.fc3(h))
        x_hat = self.fc4(h)
        return x_hat

    def encode(self, x):
        # encoder
        h = self.relu(self.fc1(x))
        h = self.fc2(h)
        return h

    def decode(self, h):
        # decoder
        h = self.relu(self.fc3(h))
        x_hat = self.fc4(h)
        return x_hat


class DeepMINEModel(nn.Module):

    def __init__(self, n_users, n_items, img_size, review_vocab_size, desc_vocab_size):
        super().__init__()

        ### parameters ###
        self.n_users  = n_users
        self.n_items = n_items

        # preference dimensions
        self.dim_u = 20
        self.dim_theta = 20

        # autoencoders dimensions
        self.dim_node = 100

        # regularization hyparameters
        self.lambda_m = 1/40
        self.lambda_d = 1/desc_vocab_size
        self.lambda_r = 1/review_vocab_size
        self.lambda_theta = 0.1
        self.lambda_beta = 0.001
        self.lambda_Wfu = 0.001

        # lambda parameters
        self.lambda_w = 1/1000
        self.lambda_b = 0
        self.lambda_q = 1/4000
        self.lambda_c = 0
        self.lambda_n = 1/4000
        self.lambda_t = 0

        ### weights ###
        # user perception factors
        self.u_users = nn.Embedding(self.n_users, self.dim_u)
        self.theta_users = nn.Embedding(self.n_users, self.dim_theta)

        # hidden information factor
        self.v_items = nn.Embedding(self.n_items, self.dim_u)

        # cognition factors
        self.a_1 = nn.Embedding(self.n_users, 1)
        self.a_2 = nn.Embedding(self.n_users, 1)
        self.a_3 = nn.Embedding(self.n_users, 1)

        # embedding layer of information integration
        self.embeddings = nn.Embedding(self.dim_node*3, self.dim_theta)

        # biases
        self.alpha_users = nn.Embedding(self.n_users, 1)
        self.beta_items = nn.Embedding(self.n_items, 1)

        # autoencoders
        self.images_autoencoder = ImageAutoEncoder(img_size=img_size, n_node=self.dim_node)
        self.descriptions_autoencoder = TextAutoEncoder(vocab_size=desc_vocab_size, n_node=self.dim_node)
        self.reviews_autoencoder = TextAutoEncoder(vocab_size=review_vocab_size, n_node=self.dim_node)

    def prediction(self,
                   user_id: int,
                   item_id: int,
                   m3,
                   d2,
                   r2,
                   ui_latent_factor,
                   ui_content_factor,
                   vj_hidden_info,
                   ui_bias,
                   ji_bias):
        ### user's cognitive styles ###
        ai1 = self.a_1(user_id)
        ai2 = self.a_2(user_id)
        ai3 = self.a_3(user_id)

        # softmax?
        # styles = F.softmax(torch.cat([ai1, ai2, ai3], dim=1), dim=1)
        # ai1, ai2, ai3 = styles[:, 0].unsqueeze(-1), styles[:, 1].unsqueeze(-1), styles[:, 2].unsqueeze(-1)

        ### information integration ###
        f_c = torch.cat([ai1.T.mm(m3), ai2.T.mm(d2), ai3.T.mm(r2)], dim=-1)
        f_j = f_c.mm(self.embeddings.weight)

        # preference
        # x_ij = ui_bias + ji_bias + ui_latent_factor.mm(vj_hidden_info.T) + ui_content_factor.mm(f_j.T)
        x_ij = ui_bias + ji_bias + (ui_latent_factor * vj_hidden_info).sum(dim=1).unsqueeze(-1) + ui_content_factor.mm(f_j.T)

        return x_ij

    def information_representation(self, image, description, review):
        # latent space
        m3 = self.images_autoencoder.encode(image)
        d2 = self.descriptions_autoencoder.encode(description)
        r2 = self.reviews_autoencoder.encode(review)

        # reconstructions
        image_hat = self.images_autoencoder.decode(m3)
        description_hat = self.descriptions_autoencoder.decode(d2)
        review_hat = self.reviews_autoencoder.decode(r2)

        return {
            'embeddings': [m3, d2, r2],
            'reconstructions': [image_hat, description_hat, review_hat]
        }

    def forward(self, user_id, p_item, p_image, p_description, p_review, n_item, n_image, n_description, n_review):
        ### information representation ###
        p_representations = self.information_representation(p_image, p_description, p_review)
        n_representations = self.information_representation(n_image, n_description, n_review)

        # latent spaces
        p_m3, p_d2, p_r2 = p_representations['embeddings']
        n_m3, n_d2, n_r2 = n_representations['embeddings']

        # reconstructions
        p_image_pred, p_description_pred, p_review_pred = p_representations['reconstructions']
        n_image_pred, n_description_pred, n_review_pred = n_representations['reconstructions']

        ### calculate user preference ###
        ui_latent_factor = self.u_users(user_id)
        ui_content_factor = self.theta_users(user_id)

        # hidden information
        p_vj_hidden_info = self.v_items(p_item)
        n_vj_hidden_info = self.v_items(n_item)

        # biases
        ui_bias = self.alpha_users(user_id)
        p_ji_bias = self.beta_items(p_item)
        n_ji_bias = self.beta_items(n_item)

        ### predictions ###
        x_ij = self.prediction(user_id, p_item, p_m3, p_d2, p_r2, ui_latent_factor, ui_content_factor, p_vj_hidden_info, ui_bias, p_ji_bias)
        x_ik = self.prediction(user_id, n_item, n_m3, n_d2, n_r2, ui_latent_factor, ui_content_factor, n_vj_hidden_info, ui_bias, n_ji_bias)

        ### losses ###
        # information representation
        l1 = torch.nn.functional.mse_loss(p_image_pred, p_image)
        l2 = torch.nn.functional.mse_loss(p_description_pred, p_description)
        l3 = torch.nn.functional.mse_loss(p_review_pred, p_review)

        # preference
        outputs = (x_ij - x_ik).squeeze()
        log_likelihood = torch.nn.functional.logsigmoid(outputs.unsqueeze(0)).sum()

        loss = -log_likelihood + l1 + l2 + l3

        return loss, outputs, (-log_likelihood, l1 + l2 + l3)

    def recommend(self, user_id, item_id, image, description, review):
        ### information representation ###
        representations = self.information_representation(image, description, review)

        # latent spaces
        m3, d2, r2 = representations['embeddings']

        ### calculate user preference ###
        ui_latent_factor = self.u_users(user_id)
        ui_content_factor = self.theta_users(user_id)

        # hidden information
        vj_hidden_info = self.v_items(item_id)

        # biases
        ui_bias = self.alpha_users(user_id)
        ji_bias = self.beta_items(item_id)

        ### predictions ###
        x_ij = self.prediction(user_id, item_id, m3, d2, r2, ui_latent_factor, ui_content_factor, vj_hidden_info, ui_bias, ji_bias)

        return x_ij