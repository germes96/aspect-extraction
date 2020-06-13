from metroide.geprepro import PhraseParser
from metroide.embending import Embedding
from metroide.distribution import Distribution
import numpy as np
import xml.etree.cElementTree as et
import pickle
from keras.preprocessing import sequence

class Parser:
    def __init__(self, params):
        self.ds_name = params.ds_name
        self.is_train = params.train
        print("Data parser initialization")
        return

    # lecture de la liste des phrase
    def getvalueofnode(self, node):
        """ return node text or None """
        return node.text if node is not None else None

    def getSentence(self):
        sentences = []
        opinions = []
        numAspect = []
        numberOfOne = 0
        numberOftwo = 0
        numberOFTree = 0
        numberOfZero = 0
        numberOfNull = 0
        numberAspects = 0
        if self.is_train == 1:
            parsedXML = et.parse("datas/"+self.ds_name +"_Train.xml")
        else:
            parsedXML = et.parse("datas/"+self.ds_name +"_Test.xml")
        for sentence in parsedXML.getroot():
            # extraction des phases
            sentences.append(self.getvalueofnode(sentence.find('text')))
            # recuperation de la liste d'opignion
            newOp = []
            numberAspect = 0
            if sentence.find('aspectTerms') is not None:
                for opinion in sentence.find('aspectTerms'):
                    newOp.append(opinion.attrib.get('term'))
                    numberAspect = numberAspect + 1
                    numberAspects = numberAspects + 1
                    if opinion.attrib.get('term') == "NULL":
                        numberOfNull = numberOfNull + 1
            else:
                newOp.append(None)
            if len(np.unique(newOp)) == 1 and newOp[0] == None:
                numAspect.append(0)
            else:
                numAspect.append(len(np.unique(newOp)))
            opinions.append(np.unique(newOp))
            if numberAspect == 0:
                numberOfZero = numberOfZero + 1
            elif numberAspect == 1:
                numberOfOne = numberOfOne + 1
            elif numberAspect == 2:
                numberOftwo = numberOftwo + 1
            elif numberAspect == 3:
                numberOFTree = numberOFTree + 1
        return sentences, opinions

    def get_reprensentation(self, sentences, opinions):
        phraseRepresentation = []
        parser = PhraseParser()
        for sentence, opinion in zip(sentences, opinions):
            representation = parser.compile(sentence, opinion)
            phraseRepresentation.append(representation)
        vocab, inv_vocab =self.build_vocab(phraseRepresentation)
        return phraseRepresentation, vocab

    def build_vocab(self, trainset):
        """
        build vocabulary from the training set and the testing set
        :param trainset:
        :param testset:
        :return:
        """
        wid = 0
        vocab, inv_vocab = {}, {}
        for record in trainset:
            for w in record:
                if w['word'] not in vocab:
                    vocab[w['word']] = wid
                    inv_vocab[wid] = w['word']
                    wid += 1
        vocab['PADDING'] = wid
        inv_vocab[wid] = 'PADDING'
        return vocab, inv_vocab

    def getTargetExemple(self, phraseRepresentation, embedd, parser):
        Example = []
        Targets = []
        size = 0
        position= 0
        self.printProgressBar(0, len(phraseRepresentation), prefix='Progress:', suffix='Complete', length=50)
        for sentence in phraseRepresentation:
            oneExample = []
            oneTarget = []
            self.printProgressBar(position, len(phraseRepresentation), prefix='Progress:' + str(position) + "/" + str(len(phraseRepresentation)), suffix='Complete', length=50)
            position = position+1
            for word in sentence:
                tag = embedd.getTagVec(word[parser.TAGTITLE])
                wd = embedd.getW2v(word[parser.WORDTITLE])
                ibo = embedd.getIBO(word[parser.IBO2])
                vector = np.concatenate((wd,tag), axis=0)
                size = len(vector)
                oneExample = np.concatenate((oneExample, vector), axis=0)
                oneTarget.append(np.array(ibo))
            Example.append(oneExample)
            Targets.append(oneTarget)
        return Example, Targets, size


    def run(self, limit = 30):
        print("Data parser: embedind init")
        embedd = Embedding()
        print("Data parser: phrase parser init")
        parser = PhraseParser()
        sentences, opinions = self.getSentence()
        print("Data parser: get sentense reprensentation")
        phraseRepresentation, vocab,  = self.get_reprensentation(sentences, opinions)
        print("Data parser: embeded full init")
        # embedd.initialiseGlove(vocab)
        embedd.initilize()
        print("Data parser: map example and target")
        Example, Targets, size = self.getTargetExemple(phraseRepresentation, embedd, parser)
        print("Data parser: write data")
        input = np.array(Example)
        output = np.array(Targets)
        finalData = {}
        sentLen = limit
        wordLen = size
        maxsize = sentLen * wordLen  # 147*15
        position= 1

        reduceInput = []
        reduceOutput = []
        reduceSentence = []
        reduceRepresentation = []
        for ins, tag, sent, rep in zip(input, output, sentences, phraseRepresentation):
            if len(ins) <= maxsize:
                reduceInput.append(ins)
                reduceOutput.append(tag)
                reduceSentence.append(sent)
                reduceRepresentation.append(rep)
        input = np.array(reduceInput)
        output = np.array(reduceOutput)

        # creation du fichier contenant les donnee
        X_train1 = sequence.pad_sequences(input, maxlen=maxsize, dtype="float32", value=0)
        for rep in reduceRepresentation:
            while len(rep) < sentLen:
                wordForm = {}
                wordForm[parser.TAGTITLE] = "NONE"
                rep.insert(0, wordForm)
            if position==1:
                position = position+1

        for target in output:
            while len(target) < sentLen:
                target.insert(0, np.array(embedd.getIBO('O')))
            if position==2:
                position = position+1

        finalTarget = []
        for target in output:
            finalTarget.append(np.array(target))

        finalData['example'] = X_train1
        finalData['target'] = finalTarget
        finalData['sentences'] = reduceSentence
        finalData['representation'] = reduceRepresentation
        out = open('data_pkl/' + str(self.ds_name) +'.pkl', 'wb')
        pickle.dump(finalData, out)
        del embedd
        out.close()
        return size

    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

