def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    import itertools, nltk, string
    
    # ------ exclude candiates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    # print punct
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # print stop_words

    # print chunker
    sents = nltk.sent_tokenize(text)
    # print sents
    words = (nltk.word_tokenize(sent) for sent in sents)
    # print words
    tagged_sents = nltk.pos_tag_sents(words)
    # print tagged_sents

    # ------ tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    chunked_sents = (chunker.parse(tagged_sent) for tagged_sent in tagged_sents)
    # for chunked_sent in chunked_sents: print chunked_sent
    
    conll_tags = (nltk.chunk.tree2conlltags(chunked_sent) for chunked_sent in chunked_sents)
    # for conll_tag in conll_tags: print conll_tag 
    all_chunks = list(itertools.chain.from_iterable(conll_tags))
    print all_chunks

    # ------ join constituent chunk words into a single chunked phrase
    #for chunk in all_chunks:
    #    lambda(word, pos, chunk): chunk != 'O'
    #    print word

    #for key, group in itertools.groupby(all_chunks, lambda(word,pos,chunk): chunk != 'O'):
    #    #print key
    #    for word, pos, chunk in group:
    #        #print key, word, pos, chunk
    #        if key: 
    #            print ' '.join(word for word,pos,chunk in group).lower() 

    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]
    #print candidates
    
    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]    

if __name__ == "__main__":
    text = "Unsupervised machine learning methods attempt to discover the underlying "
    text += "structure of a dataset without the assistance of already-labeled examples. "
    text += "Supervised machine learning methods use training data to infer a function "
    text += "that maps a set of input variables called features to some desired (and known) output value."
    print extract_candidate_chunks(text)
