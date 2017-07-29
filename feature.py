# -*- coding: utf-8 -*-

def binarize(x, k):
    """ Binarize feature k

    Args:
        x (dict)
        k (str)

    Returnx:
        x (dict)
    """
    if x[k] != '':
        x[k + '_' + x[k]] = 1.
    del x[k]
    return x

def combine(x, k1, k2):
    if x[k1] == '' or x[k2] == '':
        return x
    else:
        f = '{}{}_{}{}'.format(k1, str(x[k1]), k2, str(x[k2]))
        x[f] = 1
        return x

def delete_if_exists(x, k):
    if k in x:
        del x[k]
    return x

def extract_title(name):
    if 'Mr' in name:
        k = 'Title_Mr'
    elif 'Mrs' in name:
        k = 'Title_Ms'
    elif 'Miss' in name:
        k = 'Title_Miss'
    elif 'Master' in name:
        k = 'Title_Master'
    else:
        return ''
    return k
