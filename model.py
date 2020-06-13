# import os
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import sklearn
from sklearn.model_selection import KFold

from evals import evaluate_chunk
from metroide.TestCallback import TestCallback
import tensorflow as tf
tf.get_logger().setLevel('INFO')



class GRAM_BIGRU_CRF:
    def __init__(self, params):
        self.model_full_name = params.model_name
        self.ds_name = params.ds_name
        self.num_class = 3
        self.timesteps = params.limit_len
        self.n_epoch = params.n_epoch
        self.num_input = 448
        return


    def saveModel(name, model, split_iteration):
        model_json = model.to_json()
        with open("models/" + name + str(split_iteration) + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("models/" + name + str(split_iteration) + ".h5")

    def createModel(self):
        # Network Parameters
        num_input = self.num_input  # MNIST data input (img shape: 28*28)
        timesteps = self.timesteps  # timesteps
        num_classes = self.num_class  # MNIST total classes (0-9 digits)
        model = Sequential()
        model.add(Bidirectional(GRU(50, return_sequences=True, batch_input_shape=(None, timesteps, num_input)),
                                merge_mode='sum', weights=None))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        return model

    def getMesure(self, predicts, gold):
        save = False
        # losque toute les etiquette sont identique on a 15*4 etiquete identique donc 60
        from keras.utils import np_utils
        IBOs = []
        IBOs.append(0)
        IBOs.append(1)
        IBOs.append(2)
        # IBOs.append(3)
        IBOs = np_utils.to_categorical(IBOs, 3)
        IBO = {}
        IBO['B'] = IBOs[0]
        IBO['I'] = IBOs[1]
        IBO['O'] = IBOs[2]
        # IBO['N'] = IBOs[3]

        elm = 0
        categoricalTarget = []
        for target in gold:
            oneTag = []
            for tag in target:
                if np.sum(tag == IBO['B']) == self.num_class:
                    oneTag.append(0)
                elif np.sum(tag == IBO['I']) == self.num_class:
                    oneTag.append(1)
                elif np.sum(tag == IBO['O']) == self.num_class:
                    oneTag.append(2)
                elif np.sum(tag == IBO['N']) == self.num_class:
                    oneTag.append(3)
            categoricalTarget.append(oneTag)
        categoricalTarget = categoricalTarget

        aspectListReal = []
        aspectListRealVal = []
        for tag in categoricalTarget:
            oneList = []
            oneListVal = []
            onelmt = []
            for i in range(len(tag)):
                if tag[i] == 0 or tag[i] == 1:
                    onelmt.append(i)
                else:
                    if len(onelmt) > 0:
                        oneList.append(onelmt)
                        oneListVal.append((onelmt[0], onelmt[len(onelmt) - 1]))
                        onelmt = []
            #     print(oneList,tag,"\n\n")
            aspectListReal.append(oneList)
            aspectListRealVal.append(oneListVal)
        # recherche sur le format ibo for de predict
        aspectListTarget = []
        aspectListTargetVal = []
        for tag in predicts:
            oneList = []
            oneListVal = []
            onelmt = []
            start = 0
            for i in range(len(tag)):
                if tag[i] == 0 or tag[i] == 1:
                    onelmt.append(i)
                else:
                    if len(onelmt) > 0:
                        oneList.append(onelmt)
                        oneListVal.append((onelmt[0], onelmt[len(onelmt)-1]))
                        onelmt = []
            #     print(oneList,tag,"\n\n")
            aspectListTarget.append(oneList)
            aspectListTargetVal.append(oneListVal)

        p, r, f1, _ = evaluate_chunk(test_Y=aspectListRealVal, pred_Y=aspectListTargetVal)
        print("====> precision: " + str(p) + " rappel: " + str(r) + " f-mesure " + str(f1))
        return f1, p, r

    def buildScore(self):
        allFoldScore = []
        with open("results/" + self.model_full_name + ".txt") as f:
            for line in f:
                i = 0
                foldScore = []
                for x in line.split():
                    if i == 3 or i == 5 or i == 7:
                        foldScore.append(float(x))
                    i = i + 1
                #             print(foldScore)
                allFoldScore.append(foldScore)
            f.close()
        precisions = sum(self.column(allFoldScore, 0)) / 5
        rappel = sum(self.column(allFoldScore, 1)) / 5
        fmesures = sum(self.column(allFoldScore, 2)) / 5
        split_fold = 0
        self.readResult(fmesures, precisions, rappel)

    def readResult(self, f1, p, r):
        file = file = open("results/" + self.model_full_name + ".txt", "a+")
        file.write("fold: " + str(self.split_iteration) + " precision: " + str(p) + " rappel: " + str(r) + " f-mesure " + str(f1))
        #     file.write("fold: " + str(split_fold))
        file.write("\n")
        file.close()
        print("write file okay")

    def column(self,matrix, position):
        return [row[position] for row in matrix]

    def dataset(self, name):
        # read python dict back from the file
        pkl_file = open('data_pkl/'+name + '_Train.pkl', 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        return np.array(data["example"]), np.array(data["target"])

    #size est lier au nombre de paramettre en entree
    def run(self, size):
        self.num_input = size
        x, y, = self.dataset(str(self.ds_name))
        # x_g, y_g = self.dataset(str(self.ds_name + '_Test'))
        x, y = sklearn.utils.shuffle(x, y)

        print("x_len ", len(x))
        print("x_sahe ", x.shape)
        print("y ", len(y))
        print("y ", np.array(y).shape)

        fold_number = 5
        kfold = KFold(fold_number, True, 1)
        split = kfold.split(y)
        results = []
        self.split_iteration = 1  # l'iteration dans la boucle
        for x_train, x_test in split:
            self.trainX = x[x_train]
            self.trainY = y[x_train]
            self.testX = x[x_test]
            self.testY = y[x_test]
            self.model = self.createModel()
            sgd = optimizers.SGD(lr=0.07)
            # self.model.compile( optimizer='adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            best_weights_filepath = './best_weights.hdf5'
            saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                            mode='auto')
            batch_size = int(len(self.trainX))
            batch_size_test = int(len(self.testY))
            batch_x = np.array(self.trainX).reshape((batch_size, self.timesteps, self.num_input))
            batch_y = self.trainY
            self.test_x = np.array(self.testX).reshape((batch_size_test, self.timesteps, self.num_input))
            self.test_y = self.testY


            sendparams = self
            history = self.model.fit(batch_x,
                                batch_y,
                                epochs= self.n_epoch,
                                verbose=1,
                                shuffle=True,
                                batch_size=50,
                                # validation_split=0.1,
                                callbacks=[saveBestModel,
                                           TestCallback(sendparams,(self.test_x, self.test_y),(batch_x, batch_y))])
            self.model.summary()
            model_name = "models/" + self.model_full_name + str(self.split_iteration) + '.h5'
            self.model.load_weights(model_name)
            predict = self.model.predict_classes(self.test_x, verbose=1)
            scores = self.model.evaluate(self.test_x, self.test_y, verbose=1)
            print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
            f1, pre, rap = self.getMesure(predict, self.test_y)
            result = {}
            result['f1'] = f1
            result['p'] = pre
            result['r'] = rap
            results.append(result)
            self.split_iteration += 1
        precision = 0
        rapell = 0
        f_mesure = 0
        for result in results:
            f_mesure = f_mesure + result['f1']
            precision = precision + result['p']
            rapell = rapell + result['r']
        print("End ====> precision: " + str(precision/fold_number) + " rappel: " + str(rapell/fold_number) + " f-mesure " + str(f_mesure/fold_number))







