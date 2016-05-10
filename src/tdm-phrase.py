#! /usr/bin/python

import itertools, nltk, string
import numpy as np 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from mylsa import *
from string import maketrans

class PhraseDocMatrix(object):

    def __init__(self, stop_words, ignore_chars):
        self.stop_words = stop_words
        self.ignore_chars = ignore_chars
        self.pdict = {}
        self.dcount = 0
        #self.grammer = r'TK: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
        self.grammer = r'TK: {<JJ>* <NN.*>+}'
        self.stemmer = WordNetLemmatizer()
        
    def parseDoc(self, doc):
        terms = self.extractTerms(doc) 
        for term in terms:
            if term in self.pdict:
                self.pdict[term].append(self.dcount)
            else:
                self.pdict[term] = [self.dcount]
            '''
            words = term.split()
            if len(words) > 1: 
                for w in words:
                    try:
                        w = str(w).lower().translate(None, self.ignore_chars)
                        if w in self.stop_words:
                            continue
                        elif len(w) <= 2:
                            continue
                        elif w[0].isdigit():
                            continue
                        elif w.startswith('http'):
                            continue
                        elif w in self.pdict:
                            self.pdict[w].append(self.dcount)
                        else:
                            self.pdict[w] = [self.dcount]
                    except ValueError:
                        print "igonre term : " + w
            '''
        self.dcount += 1

    def extractTerms(self, doc):
        #doc = lambda doc: doc.decode('utf8', 'ignore')
        doc = doc.decode('utf-8')
        sents = nltk.sent_tokenize(doc)
        words = (nltk.word_tokenize(sent) for sent in sents)
        tagged_sents = nltk.pos_tag_sents(words)
        chunker = nltk.chunk.regexp.RegexpParser(self.grammer)
        chunked_sents = (chunker.parse(tagged_sent) for tagged_sent in tagged_sents)
        conll_tags = (nltk.chunk.tree2conlltags(chunked_sent) for chunked_sent in chunked_sents)
        all_chunks = list(itertools.chain.from_iterable(conll_tags))
        candidates = [' '.join(word for word, pos, chunk in group).lower()
                for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]
        terms = [cand for cand in candidates
                if cand not in self.stop_words and not all(char in self.ignore_chars for char in cand)]
      
        result = [] 
        for term in terms:
            phrase = self.stemming(term)
            result.append(phrase)
            '''
            words = phrase.split()            
            if len(words) > 1:
                for w in words:
                    try:
                        w = str(w).lower().translate(None, self.ignore_chars)
                        if w in self.stop_words:
                            continue
                        elif len(w) <= 2:
                            continue
                        elif w[0].isdigit():
                            continue
                        elif w.startswith('http'):
                            continue
                        else:
                            result.append(w)
                    except ValueError:
                        print "igonre term : " + w
            '''
        return result

    def stemming(self, term):
        words = term.split()
        result = ""
        try:
            for i, word in enumerate(words):
                if i != 0:
                    result += " " + self.stemmer.lemmatize(word)
                else:
                    result += self.stemmer.lemmatize(word)
        except ValueError:
            print "ignore " + term
            return ""
        return result

    def build(self):
        self.keys = [k for k in self.pdict.keys() if len(self.pdict[k]) > 1]
        #self.keys = [k for k in self.pdict.keys()]
        self.keys.sort()
        #for key in self.keys: print key
        # create a matrix of words by document
        self.A = np.zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.pdict[k]:
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
            words = term.split()
            if len(words) > 1:
                for i, key in enumerate(self.keys):
                    if term == key:
                        vector[i] += 1
                    for word in words:
                        if word == key:
                            vector[i] += 1                    
            else:                
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
    ignore_chars = string.punctuation
    corpus = "./geothermal-samples.txt"
    corpus = "./corpus2.txt"
    corpus = "./gmd-corpus.txt"
    pdm = PhraseDocMatrix(stop_words1, ignore_chars)
    with open(corpus) as f:
        for counter, line in enumerate(f):
            pdm.parseDoc(line.rstrip('\n'))
            if counter >= 2000:
                break

    logging = False

    pdm.build()            
    if logging : pdm.printA("---------- orginal matrix -----------")

    mylsa = LSA(pdm.getMatrix(), pdm.getTerms())

    mylsa.tfidf()
    if logging : mylsa.printA("---------- tfidf   matrix -----------")


    mylsa.SVD()
    rows = mylsa.autoSvdSize()
    if logging : print "---------- svd matrix size-----------\n" + str(rows)
    if logging : mylsa.printSVD(rows)

    mylsa.buildSimilarityMatrix(rows)
    if logging : mylsa.printSimilarityMatrix()

    '''
    mylsa.cosineSimilarity()
    if log : mylsa.printCosineSimilarity()
    mylsa.sortedCosineSimilarity()
    '''

    with open("./sql.txt") as f:
        for i,line in enumerate(f):
            if i > 0: break;
            query_vector = pdm.buildQueryVector(line.rstrip("\n"))
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
