import logging
import stanfordcorenlp
from  stanfordcorenlp import StanfordCoreNLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from keras.utils import np_utils
class PhraseParser:
    def getGoodRole(self):
        return self.good_role

    def __init__(self):
        self.nlp = StanfordCoreNLP('http://localhost', port=9000)
        self.NOMINALLIST = ["NN", "NNP", "NNPS", "NNS","FW","CD"]
        # self.good_role = ['root', 'nsubj', 'dobj', 'case', 'nmod', 'cc', 'conj', 'xcomp', 'det', 'mwe', 'amod', 'compound', 'punct', 'aux', 'advmod', 'neg', 'ccomp', 'mark', 'nmod:poss', 'cop', 'acl:relcl', 'nummod', 'acl', 'dep', 'appos', 'compound:prt', 'auxpass', 'advcl', 'nmod:tmod', 'parataxis', 'nsubjpass', 'discourse', 'expl', 'csubj', 'det:predet', 'nmod:npmod', 'iobj', 'cc:preconj', 'csubjpass']
        self.good_role = ["acomp","advcl","advmod","agent","amod","appos","aux","auxpass","cc","ccomp","conj","cop","csubj","csubjpass","dep"
            ,"det","discourse","dobj","expl","goeswith"
            ,"iobj","mark","mwe","neg","nn"
            ,"npadvmod","nsubj","nsubjpass","num","number"
            ,"parataxis","pcomp","pobj","poss","possessive"
            ,"preconj","predet","prep","prepc","prt"
            ,"punct","quantmod","rcmod","ref","root"
            ,"tmod","vmod","xcomp","xsubj","compound"
            ,"case","nmod","nmod:poss","acl:relcl","nummod","acl","compound:prt","nmod:tmod","det:predet","nmod:npmod","cc:preconj",""]
        self.good_rel = ['NN-nsubj-NN', 'NNS-dobj-VBG', 'NN-nmod-NNS', 'NN-nmod-NN', 'RB-advmod-VB', 'IN-mark-NN', 'NN-appos-NN', 'NN-compound-NNS', 'TO-mark-VB', 'NNP-compound-NNP', 'DT-neg-NN', 'PDT-det:predet-NNS', 'JJ-amod-NNP', 'JJS-amod-NN', 'CD-nummod-NNS', 'NN-compound-NNP', 'NNS-compound-NN', 'JJ-amod-NNS', 'NNS-nmod-NN', 'NNP-compound-NN', 'JJ-amod-NN', 'PRP$-nmod:poss-NN', 'NNP-compound-NNS', 'VBD-cop-NN', 'VBN-amod-NN', 'RB-advmod-NN', 'RB-advmod-VBN', 'JJR-amod-NN', 'NN-dobj-VB', 'CD-nummod-NNP', 'NNP-nmod-NN', 'PRP$-nmod:poss-NNS', 'PRP-nsubj-VBZ', 'PRP-nsubj-NN', 'NN-compound-NN', 'CD-nummod-NN', 'VBZ-cop-NN', 'NN-nsubj-VBZ', 'RB-advmod-VBZ', 'NN-nmod-VB']
        self.WORDTITLE = "word"
        self.WORDSENTENCE = "sentence"
        self.TAGTITLE = "tag"
        self.FIRSTNOUNTITLE = "first_noun"
        self.INGROUPTITLE = "in_group"
        self.GROUPETEXT = "group_text"
        self.ASPECT_NUMBER = "aspect_number"
        self.ROLE = "role"
        self.REL = "rel"
        self.TARGETAG = "target_tag"
        self.IBO2 = "ibo"

    def compile(self ,text="" , aspects=""):
        self.text = text
        self.aspects = aspects
        stop_words = stopwords.words('english')
        tokenizer = RegexpTokenizer(r'\w+')
        # tokenizer = self.nlp.word_tokenize(r'\w+')
        # tokenizer = RegexpTokenizer('')
        # text = "Service here was great, food was fantastic."
        # text = "Chow fun was dry; pork shu mai was more than usually greasy and had to share a table with loud and rude family."
        tokenized_words = tokenizer.tokenize(self.text)
        tokenized_words = self.nlp.word_tokenize(self.text)
        # goodSentence = []
        # [goodSentence.append(word) for word in tokenized_words if word not in stop_words]
        sentence = ' '.join(tokenized_words)
        #print(sentence)
        # print(self.nlp.parse(text))
        sentenseTags = self.nlp.pos_tag(sentence)
        sentenseRepresentation = []
        # TODO Recuperation du mot et de son tag dans la phrase
        for tagger in sentenseTags:
            wordForm = {}
            wordForm[self.WORDTITLE] = tagger[0]
            wordForm[self.WORDSENTENCE] = sentence
            wordForm[self.TAGTITLE] = tagger[1]
            sentenseRepresentation.append(wordForm)
        # TODO RECUPERATION DU DEBUT DU GROUPE NOMINAL
        BEGIN = []
        BEGIN.append(True)
        BEGIN.append(False)
        isFirst = False
        # BEGIN =np_utils.to_categorical(BEGIN, len(BEGIN))
        for elm in sentenseRepresentation:
            if elm[self.TAGTITLE] in self.NOMINALLIST:
                if isFirst == False:
                    elm[self.FIRSTNOUNTITLE] = BEGIN[0]
                    isFirst = True
                else:
                    elm[self.FIRSTNOUNTITLE] = BEGIN[1]
            else:
                elm[self.FIRSTNOUNTITLE] = BEGIN[1]
                isFirst = False
        # TODO DES MOTS CONTENU DANS UN GROUPE DE MOTS ASSOCIER AU GROUPE
        groupInProgress = False
        oneGroupeTxt = ""  # contien la phrase qui resume le groupe
        oneGroupeList = []  # Contien les identifiant des mot qui appartiennent au meme groupe
        for i in range(len(sentenseRepresentation)):
            if sentenseRepresentation[i][self.TAGTITLE] in self.NOMINALLIST:
                if groupInProgress == False:
                    groupInProgress = True
                    sentenseRepresentation[i][self.INGROUPTITLE] = BEGIN[1]
                else:
                    sentenseRepresentation[i][self.INGROUPTITLE] = BEGIN[0]
                    sentenseRepresentation[i - 1][self.INGROUPTITLE] = BEGIN[0]
                    if i - 1 not in oneGroupeList:
                        oneGroupeList.append(i - 1)
                    if i not in oneGroupeList:
                        oneGroupeList.append(i)
                    sent = []
                    for j in oneGroupeList:
                        sent.append(sentenseRepresentation[j][self.WORDTITLE])
                    for j in oneGroupeList:
                        sentenseRepresentation[j][self.GROUPETEXT] = ' '.join(sent)
            else:
                # print(sentenseRepresentation[i][self.WORDTITLE])
                oneGroupeList = []
                sentenseRepresentation[i][self.INGROUPTITLE] = BEGIN[1]
                groupInProgress = False

        # TODO DEFINIR LA REFERENCE DU MOT
        dependency = self.nlp.dependency_parse(sentence)
        i = 0
        # [print (elmt) for elmt  in sentenseRepresentation]
        existingDep = []

        for dependance in dependency:
            role = (dependance[0]).lower()
            cible = dependance[1]-1
            mot = dependance[2]-1
            full_role = str(sentenseTags[mot][1]) + "-" + role + "-" + str(sentenseTags[cible][1])
            if mot>=0 and role in self.good_role:
                sentenseRepresentation[cible][self.ROLE] = role
                sentenseRepresentation[cible][self.TARGETAG] = sentenseRepresentation[mot][self.TAGTITLE]
                existingDep.append(cible)
            else:
                print(role)

        for i in range(len(sentenseRepresentation)):
            if i not in existingDep:
                sentenseRepresentation[i][self.ROLE] = None
                sentenseRepresentation[i][self.TARGETAG] = None

        # for dependance in dependency:
        #     if i > 0:
        #         existingDep.append(dependance[2] - 1)
        #         if dependance[0] in self.good_role:
        #             # dependance[2]
        #             sentenseRepresentation[dependance[2] - 1][self.ROLE] = dependance[0]
        #             sentenseRepresentation[dependance[2] - 1][self.TARGETAG] = sentenseRepresentation[dependance[1] - 1][
        #                 self.TAGTITLE]
        #         else:
        #             sentenseRepresentation[dependance[2] - 1][self.ROLE] = None
        #             sentenseRepresentation[dependance[2] - 1][self.TARGETAG] = None
        #     i = i + 1
        # for i in range(len(sentenseRepresentation)):
        #     if i not in existingDep:
        #         sentenseRepresentation[i][self.ROLE] = None
        #         sentenseRepresentation[i][self.TARGETAG] = None

        # TODO ETIKETAGE O FORMAT IBO
        existTruePositif = []
        for aspect in self.aspects:
            begin = True
            # print("debut",begin)
            if aspect!=None and aspect!="NULL":
                for word in self.nlp.word_tokenize(aspect):
                    for elmt in range(len(sentenseRepresentation)):
                        # print(sentenseRepresentation[elmt][self.WORDTITLE], begin)
                        if sentenseRepresentation[elmt][self.WORDTITLE] == word and elmt not in existTruePositif:
                            existTruePositif.append(elmt)
                            if begin==True:
                                sentenseRepresentation[elmt][self.IBO2] = 'B'
                                # print("B")
                                begin = False
                            else:
                                # print("I")
                                sentenseRepresentation[elmt][self.IBO2] = 'I'
                            break
        # TODO complete with O aspect
        for elmt in range(len(sentenseRepresentation)):
            if elmt not in existTruePositif:
                sentenseRepresentation[elmt][self.IBO2] = 'O'

        #TODO SUPPRESSION DES CARACTERE NUMERIQUE
        # removalElm = []
        # for elmt in sentenseRepresentation:
        #     if elmt[self.TAGTITLE] == "CD":
        #         removalElm.append(elmt)
        # for elmt in removalElm:
        #     sentenseRepresentation.remove(elmt)

        return sentenseRepresentation