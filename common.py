

"""
Definitions that don't fit elsewhere.

"""

__all__ = (
    'DIGITS',
    'LETTERS',
    'CHARS',
    'sigmoid',
    'softmax',
)

import numpy


DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64)) #returns an array of exponets for the given array 'a'
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis] #returns an array of softmax(cross entropy) for the given array 'exps'

def sigmoid(a):
  return 1. / (1. + numpy.exp(-a)) #returns an array of sigmoid for the given array 'a'

