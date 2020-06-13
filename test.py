import numpy as np
import pickle
from keras.models import model_from_json

from evals import evaluate_chunk

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import GRU
from keras.layers import Bidirectional



class Test:
    def __init__(self, params):
        self.model_full_name = params.model_name
        self.ds_name = params.ds_name
        self.num_class = 3
        self.timesteps = 30
        self.num_input = 336
        return

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

    def dataset(self, name):
        # read python dict back from the file
        pkl_file = open('data_pkl/'+name + '_Test.pkl', 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        datas = {}
        datas["data"] = []
        datas["labels"] = []
        return np.array(data["example"]), np.array(data["target"]), np.array(data["sentences"]), np.array(
            data["representation"])


    def getMesure(self, predicts, gold):
        save = False
        # losque toute les etiquette sont identique on a 15*4 etiquete identique donc 60
        from keras.utils import np_utils
        full = self.timesteps * self.num_class
        correctClass = 0
        all = 0
        num_good = 0
        display = False
        good_pred = []
        for pred, real, sent, rep in zip(predicts, gold, self.z, self.r):
            all = all + 1
            pre = np_utils.to_categorical(pred, self.num_class)
            # print("prediction_simple", pred,"prediction" ,pre,  "realite",real, "verdict" , pre == real)
            good_pred.append(pre)
            num_good = num_good + np.sum(pre == real)
            if np.sum(pre == real) == full:
                #         print(sent,pred)
                correctClass = correctClass + 1
        #     else:
        #         print(rep)
        # print(correctClass, "/", all, " nombre de bon elmt :", num_good, "/", len(predicts) * full)

        # print(predict[32])
        # print(np_utils.to_categorical(predict[32], self.num_class))
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
        # recherche sur le format ibo for de reality
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

    def run(self):
        self.x, self.y, self.z, self.r = self.dataset(str(self.ds_name))
        self.test_x = np.array(self.x).reshape((len(self.x), self.timesteps, self.num_input))
        self.test_y = np.array(self.y).reshape((len(self.y), self.timesteps, self.num_class))
        # print(self.x.shape)
        fold_number = 5
        results = []
        loaded_model = self.createModel()
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # load weights into new model
        loaded_model.fit(self.test_x,
                         self.test_y,
                         epochs=1,
                         verbose=0,
                         shuffle=True,
                         batch_size=500)
        for i in range(fold_number):
            model_name = "models/" + self.model_full_name + str(i+1)
            # load json and create model
            print("Loaded model from disk")
            loaded_model.load_weights(model_name + ".h5")
            predict = loaded_model.predict_classes(self.test_x, verbose=1)
            f1, pre, rap = self.getMesure(predict, self.test_y)
            result = {}
            result['f1'] = f1
            result['p'] = pre
            result['r'] = rap
            results.append(result)
        precision = 0
        rapell = 0
        f_mesure = 0
        for result in results:
            f_mesure = f_mesure + result['f1']
            precision = precision + result['p']
            rapell = rapell + result['r']
        print("End ====> precision: " + str(precision / fold_number) + " rappel: " + str(
            rapell / fold_number) + " f-mesure " + str(f_mesure / fold_number))