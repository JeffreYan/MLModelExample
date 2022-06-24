import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import datetime

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix


class Model:
    def __init__(self):

        self.d_tree = tree.DecisionTreeClassifier(max_depth=5)
        self.is_fitted = False

    def train(self, users):
        users.drop(['Unnamed: 0', 'survey_followup', 'ruid'], axis=1, inplace=True)
        users.dropna(inplace=True)
        users.event_label = users.event_label.map({'activation': 1, 'cancellation': 0})
        users.event_timestamp = pd.to_datetime(users['event_timestamp'], unit='s')
        users = users.sort_values('event_timestamp').drop_duplicates('userid', keep='last')

        # create dummy variables for all survey questions except 4 as discussed in Jupyter notebook
        q1 = pd.get_dummies(users.survey_question_1, prefix="survey_question_1")
        users = users.join(q1)

        q2 = pd.get_dummies(users.survey_question_2, prefix="survey_question_2")
        users = users.join(q2)

        q3 = pd.get_dummies(users.survey_question_3, prefix="survey_question_3")
        users = users.join(q3)

        target_date = pd.to_datetime(datetime.date(2015, 7, 1))
        users['before7_15'] = users['event_timestamp'] < target_date
        users['before7_15'] = users['before7_15'].map({True: 1, False: 0})

        # create predictors dataframe for training
        users_train = users.drop(
            ['userid', 'event_timestamp', 'survey_timestamp', 'event_label', 'survey_question_1',
             'survey_question_2', 'survey_question_3'], axis=1)

        # training and test split done here
        x_train, x_test, y_train, y_test = train_test_split(users_train, users['event_label'], test_size=0.20)

        # create decision tree model
        self.d_tree = self.d_tree.fit(x_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, x_test):
        if not self.is_fitted:
            raise NotFittedError

        predictions = pd.Series(self.d_tree.predict(x_test))
        return predictions

    def evaluate(self, x_test, y_test):
        y_pred = self.d_tree.predict(x_test)
        y_test = y_test.reset_index(drop=True)

        print(confusion_matrix(y_test, y_pred))

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred))
        print("Recall:", metrics.recall_score(y_test, y_pred))

        return self