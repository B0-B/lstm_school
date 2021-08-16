#!/usr/bin/env python3
class JSON(dict):
    #dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

'''
Will make json converted dicts accessible via method chaining.

J = JSON({key: value})
J.key -> value
'''