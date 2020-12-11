#Gregory Marinakis HP12772
#The language model as described in Q4. runs with the provided wrapper script

import sys
import pickle
import math

class n_gram_model:
    def __init__(self, n=1, lmbd=0, unk_uniq=3, unk_freq=1):
        self.n = n
        self.lmbd = lmbd #laplace parameter
        self.unk_uniq = unk_uniq
        self.unk_freq = unk_freq
        self.v = 0
        self.m = 0
        self.perplex = 0
        self.trainSents = [] #all sentences in training data
        self.lexicon = {"<EOS>":0,"<UNK>":0} #lexicon L
        self.grams = {} #dictionary of n-grams and counts
        self.contexts = {} #dictionary of all existing contexts
        
    #train the model
    def train(self, path):
        if ".conllu" in path:
            self.read_conllu(path, self.trainSents, self.lexicon, True)
        elif ".txt" in path:
            self.read_txt(path, self.trainSents, self.lexicon, True)
        self.unkify(self.unk_uniq, self.unk_freq, self.trainSents, self.lexicon)
        self.find_counts(self.n, self.trainSents, self.grams, self.contexts)
        self.v = self.find_v(self.lexicon)
        self.m = self.find_m(self.contexts)

    #read in data and store as sentences and lexicon. the "sentences" array and "words" dictionary are passed by reference
    def read_conllu(self, path, sentences, words, training):
        file = open(path, "r", encoding="utf8")
        currSent = []
        #store 2d array of sentences in the file
        for i in file:
            currLine = i
            if currLine[0] == "\n":
                currSent.append("<EOS>")
                words["<EOS>"] += 1
                sentences.append(currSent)
                currSent = []
                for j in range(self.n-1):
                    currSent.append("<BOS>")
            elif currLine[0] != "#":
                #if currLine.split()[3] != 'PUNCT':
                currWord = currLine.split()[1]
                if currWord not in words: 
                    if not training and currWord not in self.lexicon: #this is specified because when training, new words should be added, but when evaluating, new words should be UNK
                        words["<UNK>"] += 1
                        currWord = "<UNK>"
                    else:
                        words.update({currWord:1})
                else:
                    words[currWord] += 1
                currSent.append(currWord)
        words.update({"<EOS>":len(sentences)})
        file.close()

    #for training the model on the two sentences in Q3 and other testing purposes only.
    def read_txt(self, path, sentences, words, training):
        file = open(path, "r", encoding="utf8")
        currSent = []
        for i in file:
            currSent = []
            for j in range(self.n-1):
                currSent.append("<BOS>")
            currLine = i.split()
            for j in currLine:
                if j not in words: 
                    if (not training) and (j not in self.lexicon): #this is specified because when training, new words should be added, but when evaluating, new words should be UNK
                        words["<UNK>"] += 1
                        j = "<UNK>"
                    else:
                        words.update({j:1})
                else:
                    words[j] += 1
                currSent.append(j)
            currSent.append("<EOS>")
            words["<EOS>"] += 1
            sentences.append(currSent)
        file.close()
        
    #unk-ify words to a fixed lexicon in training data
    def unkify(self, unk_uniq, unk_freq, sentences, words):
        UNKList = [] 
        for i in words:
            prev = ""
            count = 1
            repeatFlag = False
            for j in range(len(i)):
                if i[j] == prev:
                    count += 1
                else:
                    prev = i[j]
                    count == 1
                if count >= unk_uniq: #words with sequences of unk_uniq or more of the same character are UNKed
                    repeatFlag = True
            
            #words of frequency <= unk_freq and have a sequences of unk_uniq of the same character are UNked
            if (words[i] <= unk_freq) and repeatFlag:
                words["<UNK>"] += 1
                UNKList.append(i)
        for i in UNKList:
            del words[i]
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if sentences[i][j] not in words:
                    sentences[i][j] = "<UNK>"

    #find the frequency of all contexts and context word pairs in the text. Any context word pairs not in this list are assumed to be 0
    def find_counts(self, n, sentences, grams, contexts):
        #find n-1 pairs
        for i in range(len(sentences)):
            for j in range(len(sentences[i])-n+1):
                currkey = ""
                for k in range(j, j+n-1):
                    currkey += sentences[i][k] + " "

                #add context to dict
                if currkey[:len(currkey)-1] not in contexts: #[:len(currkey)-1] is to get rid of trailing space
                    contexts.update({currkey[:len(currkey)-1]:1})
                else:
                    contexts[currkey[:len(currkey)-1]] += 1
                #add context, word pair to grams
                currkey += sentences[i][j+n-1]
                if currkey not in grams:
                    grams.update({currkey:1})
                else:
                    grams[currkey] += 1
    
    #find v and m since ints are immutable
    def find_v(self, words):
        v = 0
        for i in words:
            v += words[i]
        return v

    def find_m(self, contexts):
        m = 0
        for i in contexts:
            m += contexts[i]
        return m
            
    
    #calculate the probability of word x given context y(y can be multiple words). n_gram is all words in y x order with spaces inbetween each word.
    def p(self, n_gram):
        if len(n_gram.split()) != self.n:
            print("Incorrect sized input")
            return 0
        else:
            count_gram = 0 if n_gram not in self.grams else self.grams[n_gram]
            count_contexts = 0 if n_gram[:n_gram.rfind(" ")] not in self.contexts else self.contexts[n_gram[:n_gram.rfind(" ")]]
            return (count_gram + self.lmbd) / (self.m + (self.v*self.lmbd))

    #return the perplexity of the given dataset
    def evaluate(self, path):
        evalSents = []
        evalLex = {"<EOS>":0,"<UNK>":0}
        evalGrams = {}
        evalContexts = {}
        if ".conllu" in path:
            self.read_conllu(path, evalSents, evalLex, False)
        elif ".txt" in path:
            self.read_txt(path, evalSents, evalLex, False)
        self.unkify(self.unk_uniq, self.unk_freq, evalSents, evalLex)
        self.find_counts(self.n, evalSents, evalGrams, evalContexts)
        #eqn 4
        perplx = 0
        for i in evalGrams:
            prob = self.p(i)
            perplx += (0 if prob == 0 else math.log(len(evalGrams) * prob))
        m = len(evalGrams) * len(evalLex)
        perplx = math.exp(-1 * perplx / m)
        return perplx

    #print all the probabilities. for testing
    def print_all(self):
        for i in self.contexts:
            for j in self.lexicon:
                print(j,"|",i,self.p(i+" "+j))
            print("===========")

def main():
    #defaults
    name = "mle"
    n = 1
    path = "UD_English-EWT/en_ewt-ud-train.conllu"
    evalpath = "UD_English-EWT/en_ewt-ud-dev.conllu"
    storepath = "model.p"
    unk_uniq = 3
    unk_freq = 1
    lmbd = 0

    args = sys.argv
    name = args[1]
    n = int(args[2])
    path = args[3]
    evalpath = args[4]
    storepath = args[5] +"/zh_model.p"
    if len(args) > 6: #last three hyperparameters are lmbd unk_uniq unk_freq. unk_uniq and unk_freq must either both present or neither present
        if name == "laplace":
            lmbd = int(args[6])
            if len(args) > 7:
                unk_uniq = int(args[7])
                unk_freq = int(args[8])
        elif name == "mle":
            if len(args) > 7:
                unk_uniq = int(args[6])
                unk_freq = int(args[7])

    #Question 4 code to get Q3 sentences model
    #path = "mouse.txt"
    #evalpath = "mouse.txt"
    #storepath = "Q4mouse.p"
    #unk_uniq = 3
    #unk_freq = 0
    #lmbd = 1

    print("n γ perplexity")
    #Fill out the chart for Q4
    #for i in range(2):
    #    for j in range(1,4):
    #        model = n_gram_model(j, i, unk_uniq, unk_freq)
    #        model.train(path)
    #        model.perplx = model.evaluate(evalpath)
    #        print(model.n, model.lmbd, model.perplx)

    #train the model and find perplexity
    model = n_gram_model(n, lmbd, unk_uniq, unk_freq)
    model.train(path)
    model.perplx = model.evaluate(evalpath)
    print(model.n, model.lmbd, model.perplx)
    pickle.dump(model, open(storepath, "wb"))

if __name__ == "__main__":
    main()