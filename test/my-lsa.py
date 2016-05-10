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
    def __init__(self, stop_words, ignore_chars):
        self.stop_words = stop_words
        self.ignore_chars = ignore_chars
        self.wdict = {}
        self.dcount = 0
        self.stemmer = WordNetLemmatizer()
        #self.stemmer = PorterStemmer()
        #print self.stop_words
        #print self.ignore_chars
        
    def parseDoc(self, doc):
        words = doc.split();
        WordNetLemmatizer
        for w in words:
            w = w.lower().translate(None, self.ignore_chars)
            try:
                w = self.stemmer.lemmatize(w)
                #w = self.stemmer.stem(w)
            except ValueError:
                print "ignore " + w
                continue
            if w in self.stop_words:
                continue
            elif len(w) <= 2:
                continue
            elif w[0].isdigit():
                continue
            elif w.startswith('http'):
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1

    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        for key in self.keys: print key
        # create a matrix of words by document
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1
        #print self.A
        
    def tfidf(self):
        words_per_doc = sum(self.A, axis=0)
        docs_per_word = sum(asarray(self.A > 0, "i"), axis=1)
        #print words_per_doc
        #print docs_per_word
        rows, cols = self.A.shape
        for r in range(rows):
            idf = log(float(cols) / docs_per_word[r])
            for c in range(cols):
                self.A[r,c] = (self.A[r,c] / words_per_doc[c]) * idf
        #print self.A

    def SVD(self):
        self.U, self.S, self.Vt = svd(self.A)

    def printWdict(self):
        for key, val in self.wdict.iteritems():
            print key

    def printSVD(self, rows):
        print '----------------- U matrix -----------------'
        print -1*self.U[:, 0:rows]
        print '----------------- S matrix -----------------'
        print self.S
        print '----------------- V matrix -----------------'
        print -1*self.Vt[0:rows, :]

    def printWordRelation(self):
        out = open("./matrix-u.txt", 'w')
        rows, cols = self.U.shape
        for r in range(rows):
            line = self.keys[r] + "\t" + str(self.U[r,1]) + "\t" + str(self.U[r,2]) + "\n"
            out.write(line)
        out.close()

    def test(self):
        print self.U[:,1]

    def drawPlot(self):
        rows = 500
        plt.plot(-1 * self.U[0:rows:, 1], -1*self.U[0:rows, 2], 'ro')
        #plt.axis([-0.6, 0.6, -0.6, 0.6])
        for label, x, y in zip(self.keys, -1 * self.U[0:rows, 1], -1*self.U[0:rows, 2]):
            plt.annotate(label, 
                xy = (x, y), xytext = (10, 3),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                #arrowprops = dict(arrowstyle = '->', 
                #connectionstyle = 'arc3,rad=0')
                )
        plt.show()

    def printResult(self, rows):
        print '----------------- Result matrix -----------------'
        u = self.U[:, 0:rows]
        s = np.diag(self.S[0:rows])
        v = self.Vt[0:rows, :]
        r = np.dot(np.dot(u, s), v)
        self.R = r
        rows, cols = r.shape
        for i in range(rows):
            line = self.keys[i]
            for j in range(cols):
                #line += "\t" + str(np.around(r[i, j], decimals=3))
                line += "\t" + str(r[i, j])
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
    
        for i in range(cols):
            line = "D" + str(i+1)
            for j in range(cols):
                line += "\t" + str(np.around(cs[i,j], decimals=3))
            print line

        self.CS = cs

    def sotedCosineSimilarity(self):
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

if __name__ == "__main__":
    stop_words1 = nltk.corpus.stopwords.words('english')
    stop_words2 = ['and','edition','for','in','little','of','the','to']
    ignore_chars = string.punctuation
    #corpus = "./invest-sample.txt"
    corpus = "./geothermal-samples.txt"
    corpus = "./corpus2.txt"
    mylsa = LSA(stop_words1, ignore_chars)
    with open(corpus) as f:
        for counter, line in enumerate(f):
            mylsa.parseDoc(line.rstrip('\n'))
            if counter > 1000:
                break
                
    mylsa.build()
    mylsa.printA("---------- orginal matrix -----------")
    mylsa.tfidf()
    mylsa.printA("---------- tfidf   matrix -----------")
    mylsa.SVD()
    mylsa.printSVD(3)
    mylsa.printResult(3)
    mylsa.cosineSimilarity()
    mylsa.sotedCosineSimilarity()
    #mylsa.drawPlot()
