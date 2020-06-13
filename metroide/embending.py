from metroide import PhraseParser
import csv
import numpy as np
from keras.utils import np_utils
from senticnet.senticnet import SenticNet
import gensim.models.keyedvectors as word2vec
from nltk.tokenize import RegexpTokenizer


text = "Chow fun was dry; pork shu mai was more than usually greasy and had to share a table with loud and rude family."
aspects = ["Chow fun", "pork shu mai"]
sn = SenticNet()

class Embedding:
    def __init__(self):
        self.TAGMAP = {}
        self.parser = PhraseParser()
        self.readTagList()

    def initilize(self,other=None):
        self.w2v = word2vec.KeyedVectors.load_word2vec_format('utils/GoogleNews-vectors-negative300.bin', binary=True)

    def initialiseGlove(self, vocab):
        self.w2v  = self.load_embedding('data/glove.840B.300d.txt',vocab)
    #TODO LIRE LA LISTE DES TAGs
    def readTagList(self):
        with open('utils/tag.csv') as csv_file :
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                if line_count>0:
                    self.TAGMAP[row[1]]= np.zeros(36)
                    (self.TAGMAP[row[1]])[int(str(row[0]))-1] = 1
                line_count = line_count + 1
        return

    #TODO LA REPRESENTATION VECTORIEL D'UN TAG
    def getTagVec(self,tag):
        try:
            return self.TAGMAP[tag]
        except Exception:
            return np.random.uniform(-0.25, 0.25, 36)

    #TODO RECUPERATION D"UN VECTEUR POUR LES VALEURS BOOLEEN
    def getBooleanRepresentation(self,ingroup):
        groups = [True, False]
        groudVec = np_utils.to_categorical(groups,2)
        if ingroup == True:
            return groudVec[0]
        else:
            return groudVec[1]

    #TODO RECUPERATION DE LA SEMANTIQUE DU GROUPE
    def getSentics(self,word):
        sentic = np.zeros(5)
        tokenizer = RegexpTokenizer(r'\w+')
        word_token = tokenizer.tokenize(word)
        if len(word_token)>2:
            # print(tokenizer)
            word = word_token[:2]
        try:
            sentic[0] = sn.sentics(word)["pleasantness"]
            sentic[1] = sn.sentics(word)["attention"]
            sentic[2] = sn.sentics(word)["sensitivity"]
            sentic[3] = sn.sentics(word)["aptitude"]
            try:
                sentic[4] = sn.polarity_intense(word)
            except Exception:
                sentic[4] = 0
            return sentic
        except Exception:
            return np.full(5, 0)

    #TODO DU ROLE DANS LE TEXT
    def getRole (self,role):
        roles = self.parser.getGoodRole()
        if role != None:
            roleVec = []
            for elm in roles:
                roleVec.append(len(roleVec))
            rolesVector = np_utils.to_categorical(roleVec,len(roleVec))
            position = roles.index(role)
            return rolesVector[position]
        else :
            # return  np.full(len(roles), 0)
            return np.random.uniform(-0.25, 0.25, len(roles))

    def getRel(self, rel):
        relations = self.parser.getGoodRel()
        if rel != None:
            roleVec = []
            for elm in relations:
                roleVec.append(len(roleVec))
            rolesVector = np_utils.to_categorical(roleVec, len(roleVec))
            position = relations.index(rel)
            return rolesVector[position]
        else:
            return np.full(len(relations), 0)

    #TODO RECUPERATION DE LA REPRESENTATOIN W2V
    def getW2v(self,word):
        try:
            return self.w2v[word.lower()]
        except Exception:
            #print("mot: ", word.lower(), " taille: ", len(word))
            return np.random.uniform(-0.25, 0.25, 300)

    def getIBO(self, ibo):
        IBOs = []
        IBOs.append(0)
        IBOs.append(1)
        IBOs.append(2)
        IBOs = np_utils.to_categorical(IBOs, 3)
        IBO = {}
        IBO['B'] = IBOs[0]
        IBO['I'] = IBOs[1]
        IBO['O'] = IBOs[2]
        try:
            return IBO[ibo]
        except:
            return IBO['O']



    def load_embedding(self, path, vocab):
        embeddings_dict = {}
        # print(vocab)
        with open(path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in vocab:
                    # print(word.lower())
                    # print(values[1:])
                    try:
                        vector = np.asarray(values[1:], "float32")
                        embeddings_dict[word.lower()] = vector
                    except Exception:
                        # np.full(300, 0)
                        np.random.uniform(-0.25, 0.25, 300)
        return embeddings_dict

