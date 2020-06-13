from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')
from stanfordcorenlp import StanfordCoreNLP

class Distribution:
    def __init__(self, sentences):
        self.sentences = sentences
        self.stop_words = stopwords.words('english')
        self.words = {}
        self.distribution = {}
        self.nlp = StanfordCoreNLP('http://localhost', port=9000)
        self.commpile()
        # print(self.stop_words)
    def commpile(self):
        # on tokenise chaque phrase
        for sentence in self.sentences:
            tokenized_words = self.nlp.word_tokenize(sentence)
            sent = ' '.join(tokenized_words)
            sentenseTags = self.nlp.pos_tag(sent)
            for tagger in sentenseTags:
                word = tagger[0]
                tag = tagger[1]
                if word not in self.stop_words:
                    if not self.isVerb(word, tag):
                        if word in self.words:
                            self.words[word] = self.words[word]+1
                        else:
                            self.words[word] = 1
    def isVerb(self, word, tag):
        # print(word, " ", tag)
        if tag == "WRB" or tag == "UH" or tag == "PRP" or tag == "VBP" or tag == "VBG" or tag == "VB" or tag == "VBD" or tag == "VBN" or tag=="VBP" or tag == "VBZ" or tag =="." or tag == "," or tag == "CC" or tag == "RB" or tag =="RBR" or tag == "RBS" or tag == "JJ" or tag == "DT" or tag == "IN" or tag == "PRP$":
            return True
        else:
            return False
    def getSignature(self, sentence, token):
        # print(sentence + " " + token)
        tokenized_words = self.nlp.word_tokenize(sentence)
        occurence = 0
        # si le mot est un stop word alors c'est normal qui soit present dans le text
        if token in self.stop_words:
            return 0

        for word in tokenized_words:
            if word == token:
                occurence = occurence+1
        if occurence == 0:
            return 0
        else:
            if token in self.words:
                return self.words[token]/occurence
            else:
                return 0
