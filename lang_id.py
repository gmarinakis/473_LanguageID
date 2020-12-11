import pickle
import math
import random
from lm_model import n_gram_model

#randomly select between lower and upper sentences from all three languages, and save as a conllu. Change to dev for tuning, or test for testing
def generate_test(lower, upper):
    s = [[],[],[]]
    currSent = []
    file = open("UD_English-EWT/en_ewt-ud-test.conllu", "r", encoding="utf8")
    for i in file:
        currLine = i
        if currLine[0] == "\n":
            s[0].append(currSent)
            currSent = []
        else:
            currSent.append(currLine)
    file.close()
    currSent = []
    file = open("UD_Spanish-AnCora/es_ancora-ud-test.conllu", "r", encoding="utf8")
    for i in file:
        currLine = i
        if currLine[0] == "\n":
            s[1].append(currSent)
            currSent = []
        else:
            currSent.append(currLine)
    file.close()
    currSent = []
    file = open("UD_French-GSD/fr_gsd-ud-test.conllu", "r", encoding="utf8")
    for i in file:
        currLine = i
        if currLine[0] == "\n":
            s[2].append(currSent)
            currSent = []
        else:
            currSent.append(currLine)
    file.close()
    allSents = []
    correct = []
    for i in range(random.randint(lower, upper)):
        sIndex = random.randint(0,2)
        allSents.append(s[sIndex].pop(random.randint(0,len(s[sIndex])-1)))
        correct.append(sIndex)

    #write the selected senteces to a file, in case you want to repeat trial
    file = open("lang_id_test.conllu", "w", encoding="utf8")
    for i in range(len(allSents)):
        for j in range(len(allSents[i])):
            file.write(allSents[i][j])
        file.write("\n")
    file.close
    return correct
    #file = open("land_id_test_key.txt", "w")
    #for i in range(len(correct)):
    #    file.write(str(correct[i])+"\n")
    #file.write("\n")
    #file.close

#calculate precision, recall, and f1 score, and print results. also used for Q2
def evaluate(values, predict, correct):
    #Values for the 4 parts of Q2
    #print("a.")
    #values = ["T","F"]
    #correct = ["T", "T", "F", "T", "F", "T", "F", "T"]
    #predict = ["T", "F", "T", "T", "F", "F", "F", "T"]
    #print("b.")
    #values = ["T","F"]
    #correct = ["T", "F", "F", "F", "F", "F", "F", "T"]
    #predict = ["F", "T", "F", "F", "F", "F", "F", "F"]
    #print("c.")
    #values = ["T","U","F"]
    #correct = ["T", "F", "U", "F", "F", "F", "U", "T"]
    #predict = ["F", "T", "U", "F", "F", "F", "F", "T"]
    #print("d.")
    #values = ["A","B","C"]
    #correct = ["C", "C", "A", "C", "C", "C", "C", "C", "B", "A", "C", "C", "C"]
    #predict = ["C", "C", "C", "C", "C", "C", "C", "C", "B", "A", "A", "C", "C"]

    sum_p, sum_r, sum_f1, sum_tp, sum_fp, sum_fn, sum_tn = 0,0,0,0,0,0,0
    print("x TP FP FN TN A P R F1")
    wrong = [] #sentences that were judged wrong
    for i in range(len(values)):
        tp,fp,fn,tn = 0,0,0,0
        if values[i] != None: #replace condition to exclude types
            for j in range(len(predict)):
                if predict[j] == values[i]:
                    if correct[j] == values[i]:
                        tp += 1
                    else:
                        fp += 1
                        wrong.append(j)
                else:
                    if correct[j] == values[i]:
                        fn += 1
                    else:
                        tn += 1
            a = (tp+tn)/(tp+tn+fp+fn)
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            f1 = 0 if tp == 0 else (2*p*r)/(p+r)
            print(values[i], tp,fp,fn,tn,a,p,r,f1)
            sum_p += p
            sum_r += r
            sum_f1 += f1
            sum_tp += tp
            sum_fp += fp
            sum_fn += fn
            sum_tn += tn
    mac_p = sum_p/len(values)
    mac_r = sum_r/len(values)
    mac_f1 = sum_f1/len(values)
    mic_p = sum_tp/(sum_tp+sum_fp)
    mic_r = sum_tp/(sum_tp+sum_fn)
    mic_f1 = 0 if sum_tp == 0 else (2*mac_p*mac_r)/(mac_p+mac_r)
    print("Macro P,R,F1", mac_p, mac_r, mac_f1)
    print("Micro P,R,F1", mic_p, mic_r, mic_f1)
    return [mac_p, mac_r, mac_f1, mic_p, mic_r, mic_f1, wrong]

class lang_id_model():
    def __init__(self, path):
        models = []
        #monogram laplace=1 smoothed, trained using lm_model.py
        models.append(pickle.load(open("models/en_model.p", "rb")))
        models.append(pickle.load(open("models/es_model.p", "rb")))
        models.append(pickle.load(open("models/fr_model.p", "rb")))
        #models.append(pickle.load(open("models/el_model.p", "rb")))
        #models.append(pickle.load(open("models/zh_model.p", "rb")))
        sentences = []
        file = open(path, "r", encoding="utf8")
        currSent = []
        #store 2d array of sentences in the file
        for i in file:
            currLine = i
            if currLine[0] == "\n":
                currSent.append("<EOS>")
                sentences.append(currSent)
                currSent = []
            elif currLine[0] != "#":
                currWord = currLine.split()[1]
                currSent.append(currWord)
        file.close()

        self.trainSentsCount = sum(len(i.trainSents) for i in models) #total number of training sentences
        self.models = models
        self.sentences = sentences
        self.allprobs = [] #store all calculated probabilites for every sentence. for debugging

    
    def make_prediction(self):
        predict = []
        for i in range(len(self.sentences)):
            probs = [1,1,1]
            for j in range(len(self.sentences[i])):
                for k in range(len(self.models)):
                    probs[k] += math.log(self.models[k].p(self.sentences[i][j]))
            #add prior probability
            for j in range(len(probs)):
                probs[j] += (math.log(len(self.models[j].trainSents) / self.trainSentsCount))
            #find max
            predict.append(probs.index(max(probs)))
            self.allprobs.append(probs)
        return predict

def main():
    values = [0,1,2] #0 = English, 1 = Spanish, 2 = French
    correct = generate_test(500,1000)
    model = lang_id_model("lang_id_test.conllu")
    predict = model.make_prediction()
    results = evaluate(values, predict, correct)
    wrong = results[6]
    #for i in range(len(wrong)):
    #    print(model.sentences[wrong[i]], predict[wrong[i]], correct[wrong[i]])
    
    pickle.dump(model, open("lang_id_model.p", "wb"))
main()