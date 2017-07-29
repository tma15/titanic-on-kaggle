# -*- coding: utf-8 -*-
import csv
import random
import sys

import onlineml

import feature

def cross_validation(x, y, N, EPOCH):
    num_data = len(x)
    num_test = int(len(x) / N)

    avg = 0.

    for k in range(N):
        start = k * num_test
        end = (k + 1) * num_test

        model = onlineml.AveragedPerceptron()

        ids = [i for i in range(start)] + [i for i in range(end, num_data)]

        x_test = x[start:end]
        y_test = y[start:end]

        for i in range(EPOCH):
            random.shuffle(ids)
            _x_train = [x[k] for k in ids]
            _y_train = [y[k] for k in ids]
            x_train = onlineml.PairVectors(_x_train)
            y_train = onlineml.StringVectors(_y_train)

            model.fit(x_train, y_train)

        model.save('age_predictor')

        classifier = onlineml.Classifier()
        classifier.load('age_predictor')

        n_total = len(y_test)
        n_correct = 0

        for y_i, x_i in zip(y_test, x_test):
            vec = onlineml.PairVector(x_i)

            pred = classifier.predict(vec)
            pred = classifier.id2label(pred)
            if y_i == pred:
                n_correct += 1.
            else:
                print('true:', y_i, 'pred:', pred,
                      ' '.join(['{}:{}'.format(k, v) for k, v in x_i]))
        acc = float(n_correct) / n_total
        avg += acc
    avg /= float(N)
    print(avg)

def age2bin(age):
    age = float(age)
    if age < 1:
        age = '0.5'
    elif age >= 1 and age < 10:
        age = '5'
    elif age >= 10 and age < 20:
        age = '15'
    elif age >= 20 and age < 30:
        age = '25'
    elif age >= 30 and age < 40:
        age = '35'
    elif age >= 40 and age < 50:
        age = '45'
    elif age >= 50:
        age = '55'
    return age


def convert_age_fv(x):
    """Convert data to feature vector

    Args:
        x (dict)

    Returns:
        x (dict)
    """
    x = feature.delete_if_exists(x, 'PassengerId')
    x = feature.delete_if_exists(x, 'Survived')
    x = feature.delete_if_exists(x, 'Fare')

    title = feature.extract_title(x['Name'])
    if title != '':
        x[title] = 1.
    x = feature.delete_if_exists(x, 'Name')

    x = feature.binarize(x, 'Sex')
    x = feature.binarize(x, 'Ticket')
    x = feature.binarize(x, 'Pclass')
    x = feature.binarize(x, 'Parch')
    x = feature.binarize(x, 'SibSp')
    x = feature.binarize(x, 'Embarked')
    x = feature.binarize(x, 'Cabin')

    ### Delete feature that has no value
    empty = []
    for k, v in x.items():
        if v == '':
            empty.append(k)
    for k in empty:
        del x[k]

    return x

def main():
    random.seed(0)

    target = 'Age'

    x = []
    y = []

    with open(sys.argv[1], 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:

            _x = {}
            for k, v in zip(header, row):
                _x[k] = v
            _y = _x[target]
            del _x[target]

            if _y == '':
                continue
            
            _y = age2bin(_y)

            _x = convert_age_fv(_x)

            x_i = []
            for k, v in _x.items():
                if v == '':
                    continue
                x_i.append((k, float(v)))

            x.append(x_i)
            y.append(str(_y))

    EPOCH = 100
    cross_validation(x, y, 10, EPOCH)

    ids = [i for i in range(len(x))]

    model = onlineml.AveragedPerceptron()
    for i in range(EPOCH):
        random.shuffle(ids)
        _x = [x[k] for k in ids]
        _y = [y[k] for k in ids]

        x_train = onlineml.PairVectors(_x)
        y_train = onlineml.StringVectors(_y)

        model.fit(x_train, y_train)
    model.save('age_predictor')

if __name__ == '__main__':
    main()
