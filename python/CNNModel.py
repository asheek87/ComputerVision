import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg16,MobileNetV2
from tensorflow.keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras import models,layers,metrics
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV,train_test_split,KFold
import pandas as pd

class CNNModel:
    '''
    This class handles CNN mdel creation, training and parameter optimization
    '''
    metrics=['accuracy',
            metrics.Precision(name = 'Precision'),
            metrics.Recall(name = 'Recall'),
            ] 
    
    
    def __init__(self,sideDIM,XTrain,yTrain,XTest,yTest,outputNode):
        self.__sideDIM=sideDIM
        self.__X_train =XTrain 
        self.__y_train=yTrain 
        self.__X_test =XTest
        self.__y_test=yTest
        self.__seed=123
        self.__outNode=outputNode # output nodes for CNN

        self.__loss='categorical_crossentropy'
    
    def __createVGG16BaseModel(self):
        '''
        Using a pretained model VGG16
        Return base model layers

        importing VGG16 from keras with pre-trained weights that is trained on imagenet
        include_top set to False to exclude the top classification layer 
        weights is set to use the weights from pre-training on Imagenet
        '''
        base_model = vgg16.VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=(self.__sideDIM,self.__sideDIM,3)) #3 = RGB Channel

        for layer in base_model.layers:
            layer.trainable = False
        
        return base_model.layers
    
    def __createMobileNetV2BaseModel(self):
        '''
        Using a pretained model VGG16
        Return base model layers

        importing MobileNetV2 from keras with pre-trained weights that is trained on imagenet
        include_top set to False to exclude the top classification layer 
        weights is set to use the weights from pre-training on Imagenet
        '''

        base_model = MobileNetV2(weights='imagenet', 
                       include_top=False, 
                       input_shape=(self.__sideDIM,self.__sideDIM,3)) #3 = RGB Channel
        
        for layer in base_model.layers:
            layer.trainable = False
        
        return base_model

    def getLossFunction(self):
        return self.__loss

    def __createTopModel(self,anOptimizer='adam',hiddenActivation='relu',outActivation='softmax',hidUnit=256):
        # FOR VGG 16
        # baseModel=self.__createVGG16BaseModel()
        # model = tf.keras.Sequential(baseModel)
        # model.add(Flatten())
        # model.add(Dense(hidUnit, activation=hiddenActivation))
        # model.add(Dense(hidUnit, activation=hiddenActivation))
        # model.add(layers.Dropout(0.1))
        # model.add(Dense(int(hidUnit/2), activation=hiddenActivation))
        # model.add(Dense(self.__outNode, activation=outActivation)) #output layer

        baseModel=self.__createMobileNetV2BaseModel()
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x=(Dense(hidUnit, activation=hiddenActivation))(x)
        x=(Dense(hidUnit, activation=hiddenActivation))(x)
        x=(layers.Dropout(0.1))(x)
        x=(Dense(int(hidUnit/2), activation=hiddenActivation))(x)
        outputLayers=(Dense(self.__outNode, activation=outActivation))(x)

        model = models.Model(inputs=baseModel.input, outputs=outputLayers)
        
        
        model.compile(loss='categorical_crossentropy',
              optimizer=anOptimizer, 
              metrics=CNNModel.metrics
              )
        return model


    def findOptimizeParamCV(self,dictParam,fold=5):
        '''
        Return 1) dataframe for all results 2) the optmized parameter in a dictionary
        3) The best accuracy score 
        4) the model object with optimized paramter in a  dictionary

        dictModelParam: a dictionary of neural network parameters that is being tested.
        '''

        resultsCols=['params','mean_test_score','std_test_score','rank_test_score']
        
        kf = KFold(n_splits=fold,shuffle= True, random_state=self.__seed)
        self.__kFold=kf
        newtworkModel = KerasClassifier(build_fn=self.__createTopModel, epochs=20, batch_size=25, verbose=0)
        clf = GridSearchCV(estimator = newtworkModel, param_grid=dictParam,cv=kf)
        result=clf.fit(self.__X_train,self.__y_train)
   
        dfFull=pd.DataFrame(result.cv_results_)
        dfResult=dfFull[resultsCols]
        dfResult=dfResult.sort_values(by=['rank_test_score'], ascending=True)
        dfFull=dfFull.sort_values(by=['rank_test_score'], ascending=True)
        
        return dfFull,dfResult,clf.best_params_,clf.best_score_,clf.best_estimator_

    def getKFold(self):
        '''
        Return kFold object which has been initialised in findOptimizeParamCV() or retun None
        '''
        return self.__kFold

    def trainModel(self, bestParam, xData,yData):

        kf=self.getKFold()
        network=self.__createTopModel(anOptimizer=bestParam.get('anOptimizer'),outActivation=bestParam.get('outActivation'),
                                hidUnit=bestParam.get('hidUnit'))
        dfMain = pd.DataFrame()
        adict={}
        count=0

        for train, test in kf.split(xData, yData):
            history=network.fit(xData[train], yData[train], epochs=bestParam.get('epochs'), 
                                batch_size=bestParam.get('batch_size'),shuffle=True,verbose=0)
            scores = network.evaluate(xData[test], yData[test], batch_size=bestParam.get('batch_size'),verbose=0)
            for i in range(0,len(network.metrics_names)):
                adict.update({network.metrics_names[i]:scores[i]})

            df = pd.DataFrame([adict])
            df.insert(0,'fold',count+1,True)
            count+=1
            adict.clear()
            dfMain=dfMain.append(df,ignore_index = True)
        
        return dfMain,network,history 

    
