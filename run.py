# -*- coding: utf-8 -*-
import copy
import csv
import random
import sys

import onlineml

import feature
import train_age_predictor as age_pred

def cross_validation(x, y, N, EPOCH):
    num_data = len(x)
    num_test = int(len(x) / N)

    avg = 0.

    top_k_feature = 20

    for k in range(N):
        start = k * num_test
        end = (k + 1) * num_test

        model = onlineml.AveragedPerceptron()

        ids = [i for i in range(start)] + [i for i in range(end, num_data)]

        x_test = x[start:end]
        y_test = y[start:end]

        for i in range(EPOCH):
#            random.shuffle(ids)
            _x_train = [x[k] for k in ids]
            _y_train = [y[k] for k in ids]
            x_train = onlineml.PairVectors(_x_train)
            y_train = onlineml.StringVectors(_y_train)
            model.fit(x_train, y_train)

        model.save('model')
        fs = model.enumerate_features()
        label = '1' ### survive
        w = model.get_weight(label, fs[0])
        weight = []
        for f in fs:
            weight.append(model.get_weight(label, f))

        print('=== Top-{} feature weight of survived class ==='.format(top_k_feature))
        for f, w in sorted(zip(fs, weight), key=lambda x:x[1], reverse=True)[:top_k_feature]:
            print(w, f)
        print('===============================================')

        classifier = onlineml.Classifier()
        classifier.load('model')

        n_total = len(y_test)
        n_correct = 0

        for y_i, x_i in zip(y_test, x_test):
            vec = onlineml.PairVector(x_i)

            pred = classifier.predict(vec)
            pred = classifier.id2label(pred)
            if y_i == pred:
                n_correct += 1.
            else:
                print(y_i, pred, ' '.join(['{}:{}'.format(k, v) for k, v in x_i]))
        acc = float(n_correct) / n_total
        avg += acc
    avg /= float(N)
    print(avg)


def convert(x, age_predictor):
    """Convert data to feature vector

    Args:
        x (dict)

    Returns:
        x (dict)
    """

    del x['PassengerId']

    if x['Age'] == '':
        x_age = age_pred.convert_age_fv(copy.deepcopy(x))
        vec = []
        for k, v in x_age.items():
            vec.append((k, float(v)))
        vec = onlineml.PairVector(vec)
        age_cls = age_predictor.predict(vec)
        age = age_predictor.id2label(age_cls)
        x['Age'] = age

    x = feature.combine(x, 'Sex', 'Pclass')
    x = feature.combine(x, 'Sex', 'Cabin')
    x = feature.combine(x, 'Sex', 'Embarked')
    x = feature.combine(x, 'Sex', 'Parch')
    x = feature.combine(x, 'Sex', 'SibSp')

    x = feature.combine(x, 'Pclass', 'Cabin')
    x = feature.combine(x, 'Pclass', 'Embarked')
    x = feature.combine(x, 'Pclass', 'Parch')
    x = feature.combine(x, 'Pclass', 'SibSp')

    title = feature.extract_title(x['Name'])
    if title != '':
        x[title] = 1.
    del x['Name']

    if x['Age'] != '':
        x['Age'] = age_pred.age2bin(x['Age'])

    x = feature.combine(x, 'Age', 'Sex')

    x = feature.binarize(x, 'Age')
    x = feature.binarize(x, 'Sex')
    x = feature.binarize(x, 'Ticket')
    x = feature.binarize(x, 'Cabin')
    x = feature.binarize(x, 'Embarked')
    x = feature.binarize(x, 'Pclass')
    x = feature.binarize(x, 'Parch')
    x = feature.binarize(x, 'SibSp')


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
    target = 'Survived'

    x = []
    y = []

    age_predictor = onlineml.Classifier()
    age_predictor.load('age_predictor')

    with open(sys.argv[1], 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:

            _x = {}
            for k, v in zip(header, row):
                _x[k] = v
            _x = convert(_x, age_predictor)

            _y = _x[target]
            del _x[target]

            x_i = []
            for k, v in _x.items():
                if v == '':
                    continue
                x_i.append((k, float(v)))

            x.append(x_i)
            y.append(str(_y))

    EPOCH = 1000
    cross_validation(x, y, 10, EPOCH)

    x = onlineml.PairVectors(x)
    y = onlineml.StringVectors(y)

    model = onlineml.AveragedPerceptron()

    for i in range(EPOCH):
        model.fit(x, y)

    model.save('model')

    classifier = onlineml.Classifier()
    classifier.load('model')

    with open('submission.csv', 'w') as f:
        writer = csv.writer(f)

        writer.writerow(['PassengerId', 'Survived'])

        with open(sys.argv[2], 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:

                x = {}
                for k, v in zip(header, row):
                    x[k] = v
                _id = x['PassengerId']

                x = convert(x, age_predictor)

                vec = []
                for k, v in x.items():
                    vec.append((k, float(v)))
                vec = onlineml.PairVector(vec)

                pred = classifier.predict(vec)
                pred = classifier.id2label(pred)

                writer.writerow([_id, pred])

if __name__ == '__main__':
    main()
