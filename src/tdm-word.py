#! /usr/bin/python

import itertools, nltk, string
import numpy as np 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from mylsa import *

class WordDocMatrix(object):
    def __init__(self, stop_words, ignore_chars):
        self.stop_words = stop_words
        self.ignore_chars = ignore_chars
        self.wdict = {}
        self.dcount = 0
        self.stemmer = WordNetLemmatizer()
        
    def parseDoc(self, doc):
        words = doc.split();
        WordNetLemmatizer
        terms = self.extractTerms(doc)
        for w in terms:
            if w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]

        self.dcount += 1

    def extractTerms(self, doc):
        words = doc.split();
        terms = []
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
            
            terms.append(w)

        return terms
        

    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        # for key in self.keys: print key
        # create a matrix of words by document
        self.A = np.zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1
   
    def getMatrix(self):
        return self.A

    def getTerms(self):
        return self.keys
 
    def printA(self, title):
        print title
        print self.A

    def buildQueryVector(self, sql):
        terms = self.extractTerms(sql)
        rows, cols = self.A.shape
        vector = np.zeros(rows)
        for term in terms:
            for i,key in enumerate(self.keys):
                if term == key:
                    vector[i] += 1
     
        docs_per_word = sum(asarray(self.A > 0, "i"), axis=1)
        for i in range(rows):
            if vector[i] > 0:
                vector[i] = (float(vector[i]) / len(terms)) * log(float(cols) / docs_per_word[i])
        return vector
                 

if __name__ == "__main__":
    stop_words1 = nltk.corpus.stopwords.words('english')
    stop_words2 = ['and','edition','for','in','little','of','the','to']
    ignore_chars = string.punctuation
    corpus = "./geothermal-samples.txt"
    corpus = "./corpus2.txt"
    corpus = "./gmd-corpus.txt"
    wdm = WordDocMatrix(stop_words1, ignore_chars)
    with open(corpus) as f:
        for counter, line in enumerate(f):
            wdm.parseDoc(line.rstrip('\n'))
            if counter >= 2000:
                break
    wdm.build()
    logging = False    

    if logging : wdm.printA("---------- orginal matrix -----------")

    mylsa = LSA(wdm.getMatrix(), wdm.getTerms())

    mylsa.tfidf()
    if logging : mylsa.printA("---------- tfidf   matrix -----------")

    mylsa.SVD()
    rows = mylsa.autoSvdSize()
    if logging : print "---------- svd matrix size-----------\n" + str(rows)
    if logging : mylsa.printSVD(rows)

    mylsa.buildSimilarityMatrix(rows)
    #if log : mylsa.printSimilarityMatrix()

    '''
    mylsa.cosineSimilarity()
    if log : mylsa.printCosineSimilarity()
    mylsa.sortedCosineSimilarity()
    '''
    
    with open("./sql.txt") as f:
        for i,line in enumerate(f):
            if i > 0: break;
            query_vector = wdm.buildQueryVector(line.rstrip("\n"))
            if logging : print query_vector
            print "---------- query sql -----------"
            print line.rstrip("\n")
            result = mylsa.query(query_vector)
            print "---------- result set -----------"
            for i in range(len(result)):
                #print result[i]
                with open(corpus) as f:
                    for counter, line in enumerate(f):
                        if result[i] == counter:
                            print str(result[i]) + "  " + line[0:line.index("ABSTRACT:")] 
                            break;
    
