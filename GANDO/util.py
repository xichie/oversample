from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
import numpy as np
from imblearn.metrics import geometric_mean_score

def StackingMethod(X, y):
    clf1 = SVC(random_state=1)
    clf2 = RandomForestClassifier(random_state=2)
    clf3 = AdaBoostClassifier(random_state=3)

    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              # use_probas=True, 类别概率值作为meta-classfier的输入
                              # average_probas=False,  是否对每一个类别产生的概率值做平均
                              meta_classifier=LogisticRegression())

    sclf.fit(X, y)
    return sclf

def evaluate(X_train, y_train, X_test, y_test):
    stack_clf = StackingMethod(X_train, y_train)
    y_pred = stack_clf.predict(X_test)

    f1_score = metrics.f1_score(y_pred, y_test)
    g_mean = geometric_mean_score(y_pred, y_test)
    auc_score = roc_auc_score(y_test, y_pred)

    return [f1_score, g_mean, auc_score]

def sampling(gan, X, y, oversample_num=4000, iter=2, epochs=500, batch_size=100):

    pos = X[y == 1]
    neg = X[y == 0]
    pos_size = pos.shape[0]
    neg_size = neg.shape[0]

    # training 
    oversample = pos
    for epoch in range(iter):
        gan.train(oversample, neg, epochs=epochs, batch_size=batch_size)
        noise = np.random.normal(0, 1, (50, gan.latent_dim))
        oversample = np.concatenate((oversample, gan.generator.predict(noise)))

    noise = np.random.normal(0, 1, (oversample_num, gan.latent_dim))  # 采样个数
    oversample = gan.generator.predict(noise)

    pos_sample = np.concatenate((oversample, pos), axis=0)
    X_sample = np.concatenate((pos_sample, neg), axis=0)
    y_sample = np.concatenate((np.ones(pos_sample.shape[0]), np.zeros(neg_size)))

    return X_sample, y_sample
def MMD(X_sample, y_sample, X_orig, y_orig):
    sampled_pos = X_sample[y_sample==1]
    pos = X_orig[y_orig==1]
    import MMD
    import torch
    from torch.autograd import Variable
    MMD_score = []
    for i in range(1000):
        idx = np.random.randint(0, sampled_pos.shape[0], size=pos.shape[0])
        X_tensor = torch.Tensor(sampled_pos[idx])
        Y_tensor = torch.Tensor(pos)
        X_var = Variable(X_tensor)
        Y_var = Variable(Y_tensor)
        MMD_score.append(MMD.mmd_rbf(X_var, Y_var))
    # print(sum(MMD_score) / len(MMD_score))
    return sum(MMD_score) / len(MMD_score)