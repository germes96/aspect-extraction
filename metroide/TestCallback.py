from keras.callbacks import Callback


class TestCallback(Callback):
    def __init__(self,params, test_data, train_data):
        self.test_data = test_data
        self.train_data = train_data
        self.last_acc = 0
        self.model_full_name = params.model_full_name
        self.model = params.model
        self.timesteps = params.timesteps
        self.num_class = params.num_class
        self.split_iteration = params.split_iteration
        self.test_y = params.test_y
        self.getMesure = params.getMesure

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        x_t, y_t = self.train_data
        #         loss, acc = self.model.evaluate(x,y,verbose=0)
        # predict_train = self.model.predict_classes(x_t, verbose=0)
        predict = self.model.predict_classes(x, verbose=1)
        # print("Train Evaluation")
        # acc_traim = self.getMesure(predict_train, y_t)
        print("Train Evaluation")
        acc, p, r = self.getMesure(predict, self.test_y)
        # print("Test Evaluation")
        # predict = self.model.predict_classes(self.batch_gold_x, verbose=1)
        # self.getMesure(predict, self.batch_gold_y)
        # evaluate_chunk(self.test_y, predict)
        #         print('\nTesting loss : {}, acc: {}\n'.format(loss, acc))
        # print(acc)
        # print(p)
        if acc != None :
            if acc > self.last_acc:
                self.last_acc = acc
                print("model saved")
                self.saveModel(self.model_full_name)
            else:
                print("no save")

    def saveModel(self, name):
        model_json = self.model.to_json()
        with open("models/" + name + str(self.split_iteration) + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("models/" + name + str(self.split_iteration) + ".h5")



    def readResult(self, f1, p, r):
        file = file = open("results/" + self.model_full_name + ".txt", "a+")
        file.write("fold: " + str(self.split_iteration) + " precision: " + str(p) + " rappel: " + str(r) + " f-mesure " + str(f1))
        #     file.write("fold: " + str(split_fold))
        file.write("\n")
        file.close()
        print("write file okay")