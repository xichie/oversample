from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import numpy as np
import tensorflow as tf
import smote_variants as sv
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, recall_score
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras.backend as K
from load_data import load_data

#  屏蔽日志输出
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import logging
from util import sampling, evaluate
import MMD
from sklearn.metrics import silhouette_score
logging.getLogger("smote_variants").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)
import sys

np.random.seed(6)

class GAN():
    def __init__(self):
        self.dim = 19                         # 数据维度
        self.latent_dim = 100                  # 噪声维度
        self.hidden_G_nodes = 250
        self.hidden_D_nodes = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='categorical_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)

        self.combined.add_loss(self.within_class_scatter(img))
        self.combined.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # 类内散度计算
    def within_class_scatter(self, x_fake):
        m_dot = K.mean(x_fake)
        loss = K.log(1 - K.mean((x_fake - m_dot) ** 2))
        return 0.01 * loss

    def build_generator(self):
        model = Sequential()

        model.add(Dense(self.hidden_G_nodes, input_dim=self.latent_dim, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.dim, activation='tanh'))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Dense(self.hidden_D_nodes, input_dim=(self.dim), activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.summary()

        img = Input(shape=(self.dim,))
        validity = model(img)

        return Model(img, validity)

    def train(self, pos, neg, epochs=500, batch_size=32):
        neg_label = np.zeros(shape=(batch_size, 3))
        neg_label[:, 0] = 1.
        pos_label = np.zeros(shape=(pos.shape[0], 3))
        pos_label[:, 1] = 1.
        gen_label = np.zeros(shape=(batch_size, 3))
        gen_label[:, 2] = 1.


        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            idx = np.random.randint(0, neg.shape[0], batch_size)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_data = self.generator.predict(noise)

            # Train the discriminator
            d_loss_neg = self.discriminator.train_on_batch(neg[idx], neg_label)     #[1, 0, 0]
            d_loss_pos = self.discriminator.train_on_batch(pos, pos_label)  #[0, 1, 0]
            d_loss_gen = self.discriminator.train_on_batch(gen_data, gen_label)  #[0, 0, 1]
            d_loss = 0.5 * np.add(d_loss_gen, 0.5 * np.add(d_loss_neg, d_loss_pos))

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (pos.shape[0], self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, pos_label)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

if __name__ == '__main__':
    data_path = './data/segment0/'
    train, test = load_data(data_path)

    history = []
    oversampled_history = []
    mmd_score = 0
    sscore = 0
    for (x_train, y_train), (x_test, y_test) in zip(train, test):
        scaler = MinMaxScaler()
        x_train_std = scaler.fit_transform(x_train) 
        x_test_std = scaler.fit_transform(x_test)

        gan = GAN()
        X_sample, y_sample = sampling(gan, x_train_std, y_train, oversample_num=330, epochs=2000, iter=3)

        all_metrics_score = evaluate(x_train_std, y_train, x_test_std, y_test)
        history.append(all_metrics_score)

        all_metrics_score = evaluate(X_sample, y_sample, x_test_std, y_test)
        oversampled_history.append(all_metrics_score)
        print(all_metrics_score)

        mmd_score += MMD(X_sample, y_sample, x_train_std, y_train)
        sscore += silhouette_score(X_sample, y_sample)

    print('f1 score, G-mean, AUC')

    print(["%.5f" % mean + " +/- " + "%.5f" % std for mean, std in
           zip(np.mean(history, axis=0), np.std(history, axis=0))], "上采样前")
    print(["%.5f" % mean + " +/- " + "%.5f" % std for mean, std in
           zip(np.mean(oversampled_history, axis=0), np.std(oversampled_history, axis=0))], "上采样后")

    print("mmd_score: %.5f" % (mmd_score / 5.))
    print("silhouette_score: %.5f" % (sscore / 5))
