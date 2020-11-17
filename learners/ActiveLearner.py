import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


class ActiveLearner:

    def __init__(self, dataset, nEstimators, name):
        '''input: dataset -- an object of class Dataset or any inheriting classes
                  nEstimators -- the number of estimators for the base classifier, usually set to 50
                  name -- name of the method for saving the results later'''

        self.dataset = dataset
        self.indicesKnown = dataset.indicesKnown
        self.indicesUnknown = dataset.indicesUnknown
        # base classification model
        self.nEstimators = nEstimators
        self.model = RandomForestClassifier(self.nEstimators, n_jobs=8)
        self.name = name

    def reset(self):

        '''forget all the points sampled by active learning and set labelled and unlabelled sets to default of the dataset'''
        self.indicesKnown = self.dataset.indicesKnown
        self.indicesUnknown = self.dataset.indicesUnknown

    def train(self):

        '''train the base classification model on currently available datapoints'''
        trainDataKnown = self.dataset.trainData[self.indicesKnown, :]
        trainLabelsKnown = self.dataset.trainLabels[self.indicesKnown, :]
        trainLabelsKnown = np.ravel(trainLabelsKnown)
        self.model = self.model.fit(trainDataKnown, trainLabelsKnown)

    def evaluate(self, performanceMeasures):

        '''evaluate the performance of current classification for a given set of performance measures
        input: performanceMeasures -- a list of performance measure that we would like to estimate. Possible values are 'accuracy', 'TN', 'TP', 'FN', 'FP', 'auc'
        output: performance -- a dictionary with performanceMeasures as keys and values consisting of lists with values of performace measure at all iterations of the algorithm'''
        performance = {}
        test_prediction = self.model.predict(self.dataset.testData)
        m = metrics.confusion_matrix(self.dataset.testLabels, test_prediction)

        if 'accuracy' in performanceMeasures:
            performance['accuracy'] = metrics.accuracy_score(self.dataset.testLabels, test_prediction)

        if 'TN' in performanceMeasures:
            performance['TN'] = m[0, 0]
        if 'FN' in performanceMeasures:
            performance['FN'] = m[1, 0]
        if 'TP' in performanceMeasures:
            performance['TP'] = m[1, 1]
        if 'FP' in performanceMeasures:
            performance['FP'] = m[0, 1]

        if 'auc' in performanceMeasures:
            test_prediction = self.model.predict_proba(self.dataset.testData)
            test_prediction = test_prediction[:, 1]
            performance['auc'] = metrics.roc_auc_score(self.dataset.testLabels, test_prediction)

        return performance


class ActiveLearnerRandom(ActiveLearner):
    '''Randomly samples the points'''

    def selectNext(self):
        self.indicesUnknown = np.random.permutation(self.indicesUnknown)
        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([self.indicesUnknown[0]])]));
        self.indicesUnknown = self.indicesUnknown[1:]


class ActiveLearnerUncertainty(ActiveLearner):
    '''Points are sampled according to uncertainty sampling criterion'''

    def selectNext(self):
        # predict for the rest the datapoints
        unknownPrediction = self.model.predict_proba(self.dataset.trainData[self.indicesUnknown, :])[:, 0]
        selectedIndex1toN = np.argsort(np.absolute(unknownPrediction - 0.5))[0]
        selectedIndex = self.indicesUnknown[selectedIndex1toN]

        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([selectedIndex])]))
        self.indicesUnknown = np.delete(self.indicesUnknown, selectedIndex1toN)