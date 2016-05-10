#! /usr/bin/python

import os
import numpy as np
import math
from nltk.stem import PorterStemmer, WordNetLemmatizer

def list_files(path):
    files = os.listdir(path) 
    for counter, file in enumerate(files):
        print counter
        print os.path.join(path, file)
        if counter >= 9:
            return

def read_file(path):
    with open(path) as f:
        for line in f:
            print line.rstrip('\n')

def matrix_multiply():
    U = np.array([[0.22, -0.11], [0.20, -0.07], [0.24, 0.04]])
    S = np.array([[3.34, 0], [0, 2.54]]);
    V = np.array([[0.20, 0.61], [-0.06, 0.17]])
    print U
    print S
    print U * S

def diag_test():
    m = np.array([1,2,3])
    print np.diag(m)

def matrix_extend():
    a = np.array([1,2,3])
    print np.diag(a)
    print np.append(a, [4,5,6])
    print np.diag(np.append(a, [0,0,0]))

def matrix_append():
    A = np.diag(np.array([1,2,3]))
    print A
    print np.concatenate((A, [[0,0,0],[0,0,0],[0,0,0]]), axis=0)


def matrix_multiply():
    #print np.arange(9.0)
    #print [[1,2,3,4,5,6,7,8,9]].reshape((3,3))
    U = np.matrix("0.22 -0.11; 0.20 -0.07; 0.24 0.04; 0.40 0.06; 0.64 -0.17; 0.27 0.11; 0.27 0.11; 0.30 -0.14; 0.21 0.27; 0.01 0.49; 0.04 0.62; 0.03 0.45")
    S = np.matrix("3.34 0; 0 2.54")
    #V = np.matrix("0.20 0.61; -0.06 0.17; 0.11 -0.50; -0.95 -0.03; 0.05 -0.21; -0.08 -0.26; 0.18 -0.43; -0.01 0.05; -0.06 0.24")
    V = np.matrix("0.20 0.61 0.46 0.54 0.28 0.00 0.01 0.02 0.08; -0.06 0.17 -0.13 -0.23 0.11 0.19 0.44 0.62 0.53")
    print U
    print V
    print np.dot(np.dot(U, S), V)

def stem_test():
    stemmer = WordNetLemmatizer()
    print stemmer.lemmatize("environment")
    print stemmer.lemmatize("environmental")
    stemmer = PorterStemmer()
    print stemmer.stem("environmental")

def matrix_sort():
    a = np.array([[2,1,4,3,5]])
    a.sort(axis=1)
    print a

def cosineSimilarity():
    a = np.matrix([[1,1,1]])
    b = np.matrix([[2,2,2]])
    print np.dot(a, b.T)
    print np.linalg.norm(a)
    print np.linalg.norm(b)
    r1 = np.dot(a, b.T)/np.linalg.norm(a)/np.linalg.norm(b)
    print str(r1)
    r2 = np.dot(a, b.T)/np.linalg.norm(a-b)
    print str(r2)
    tmp = 0;
    for i in range(3):
        tmp += math.pow((a[0,i] - b[0,i]), 2)
    r3 = np.dot(a, b.T)/math.sqrt(tmp)
    print str(r3)


if __name__ == "__main__":
    #list_files("/home/ws/Desktop/geothermal");
    #read_file("./geothermal-samples.txt")
    #matrix_multiply()
    #matrix_extend()
    #matrix_append()
    #matrix_multiply()
    #stem_test()
    #matrix_sort()
    cosineSimilarity()
