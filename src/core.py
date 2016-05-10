#! /usr/bin/python

from numpy import zeros
from scipy.linalg import svd
from math import log
from numpy import asarray, sum
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
import itertools, nltk, string
import matplotlib.pyplot as plt
import numpy as np

class LSA(object):
    def __init__(self, td_matrix, terms):
        self.A = td_matrix
        self.T = terms
    
    def tfidf(self):
        words_per_doc = sum(self.A, axis=0)
        docs_per_word = sum(asarray(self.A > 0, "i"), axis=1)
        #print words_per_doc
        #print docs_per_word
        rows, cols = self.A.shape
        for r in range(rows):
            idf = log(float(cols) / (docs_per_word[r]))
            for c in range(cols):
                #print str(self.A[r,c]) + " / " + str(words_per_doc[c]) + "*" + str(idf)
                self.A[r,c] = (self.A[r,c] / (words_per_doc[c] + 1)) * idf
        #print self.A

    def SVD(self):
        self.U, self.S, self.Vt = svd(self.A)

    def printSVD(self, rows):
        print '----------------- U matrix -----------------'
        print -1*self.U[:, 0:rows]
        print '----------------- S matrix -----------------'
        print self.S
        print '----------------- V matrix -----------------'
        print -1*self.Vt[0:rows, :]

    def drawPlot(self, rows):
        plt.plot(-1*self.U[:, 1], -1*self.U[:, 2], 'ro')
        for label, x, y in zip(self.keys, -1*self.U[:, 1], -1*self.U[:, 2]):
            plt.annotate(label, 
                xy = (x, y), xytext = (10, 3),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                #arrowprops = dict(arrowstyle = '->', 
                #connectionstyle = 'arc3,rad=0')
                )
        plt.show()

    def buildSimilarityMatrix(self, rows):
        u = self.U[:, 0:rows]
        s = np.diag(self.S[0:rows])
        v = self.Vt[0:rows, :]
        r = np.dot(np.dot(u, s), v)
        self.R = r

    def autoSvdSize(self):
        total = 0
        for s in self.S:
            total += s

        amount = 0
        for i,s in enumerate(self.S):
            amount += self.S[i]
            if float(amount)/float(total) >= 0.8:
                return i 

    def printSimilarityMatrix(self):
        print '----------------- Result matrix -----------------'
        rows, cols = self.R.shape
        for i in range(rows):
            line = self.T[i]
            for j in range(cols):
                line += "\t" + str(np.around(self.R[i, j], decimals=3))
                #line += "\t" + str(r[i, j])
            print line

    def cosineSimilarity(self):
        print '----------------- cosine similarity -----------------'
        rows, cols = self.R.shape
        cs = zeros([cols, cols])
        for i in range(cols):
            a = self.R[:,i]
            for j in range(cols):
                b = self.R[:,j]
                cs[i,j] = np.dot(a, b.T) / np.linalg.norm(a) / np.linalg.norm(b)
        self.CS = cs

    def printCosineSimilarity(self):
        rows, cols = self.CS.shape
        for i in range(cols):
            line = "D" + str(i+1)
            for j in range(cols):
                line += "\t" + str(np.around(self.CS[i,j], decimals=3))
            print line
        

    def sortedCosineSimilarity(self):
        print '----------------- sorted doc similarity -----------------'
        rows, cols = self.CS.shape
        for i in range(rows):
            line = "D" + str(i+1)
            sorter = np.copy(self.CS[i, :])
            sorter.sort()
            for j in range(cols-2, cols-6, -1):
                line += "\t" + "D" + str(self.findIndex(self.CS[i,:], sorter[j])+1) + ": " + str(np.around(sorter[j], decimals=3))
            print line
            
            
    def findIndex(self, target, val):
        for i in range(len(target)):
            if val == target[i]:
                return i
        return -1

    def printA(self, title):
        print title
        print self.A


    def query(self, query_vector):
        rows, cols = self.R.shape
        cs = zeros(cols)
        for i in range(cols):
            b = self.R[:,i]
            cs[i] = np.dot(query_vector, b) / np.linalg.norm(query_vector - b)
            #cs[i] = np.dot(query_vector, b) / np.linalg.norm(query_vector) / np.linalg.norm(b)
        #print cs 
        sorter = np.copy(cs)
        sorter.sort();

        result = []
        for i in range(cols-1, cols-9, -1):
            result.append(self.findIndex(cs, sorter[i]))
        
        return result
 
