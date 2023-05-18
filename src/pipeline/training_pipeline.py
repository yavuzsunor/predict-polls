from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

'''
Train benchmark classification models
'''

#logistic regression
lr_clf = LogisticRegression()
lr_clf.fit(features_tr, train_labels)
print('train score:',lr_clf.score(features_tr, train_labels))
print('validation score:',lr_clf.score(features_vl, val_labels))
# need to save the model object in artifacts

#naive bayes
gnb = GaussianNB()
gnb.fit(features_tr, train_labels)
print('train score:',gnb.score(features_tr, train_labels))
print('validation score:',gnb.score(features_vl, val_labels))
# need to save the model object in artifacts

#call NN model from model_trainer to save the model object in artifacts