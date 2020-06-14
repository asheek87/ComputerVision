import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn import preprocessing

class Analyser():
    '''
    This class handles data plotting 
    '''
    def __init__(self,output1):
        self.__output1=output1

    def histogramOrBoxPlotAnalysis(self,inDataFrame,strCols=True, boxSize=(80,80), hist=True,fileName=''):
        '''
        Analyse categorical and numerical data using histogram.
        A histogram or box plot to analyse each feature column in dataframe
        Non numerical column will be converted to numerical 

        strCols: 
        If True, will include string features in histogram/boxplot diagram converted to number using label encoding
        If False, only numerical columns shown.
        Default value is True

        boxSize:
        Size of boxplot/histogram diagram. Default is 80,80

        hist:
        If true, plot histogram, else boxplot. Default is true
        '''
        dataFrame=inDataFrame.copy()
        plt.style.use('seaborn-white')
        le=preprocessing.LabelEncoder()
        cols=5
        
        numMask=dataFrame.dtypes!=np.object
        if strCols:
            dataCols=dataFrame.columns
        else:
            dataCols=dataFrame.columns[numMask]
        rows=math.ceil(len(dataCols)/cols)
        sns.set(font_scale=3.5)  # crazy big
        fig, axes = plt.subplots(rows, cols, figsize=boxSize,sharex=False)
       
        x,y,limit=0,0,0
        for aCol in dataCols:
            if dataFrame[aCol].dtypes == np.object:
                dataFrame[aCol]=le.fit_transform(dataFrame[aCol])
            if hist:
                sns.distplot(dataFrame[aCol] , color="red", ax=axes[x, y])
                imgName=fileName+'_DistPlot'+'.png'
            else:
                sns.boxplot(y=dataFrame[aCol] , color="blue", ax=axes[x, y])
                imgName=fileName+'_BoxPlot'+'.png'
            y+=1
            limit+=1
            if y%cols ==0:
                x+=1
                y=0
            if limit>=len(dataCols):
                break
        
        plt.savefig(self.__output1+imgName)
        plt.close()



    def plot_loss(self,hist,fileName=''):
        plt.style.use('seaborn-white')
        test_loss = hist.history['loss']
        epochX = list(range(1, len(test_loss) + 1))
        plt.plot(epochX, test_loss, color = 'red', label = 'test loss')
        # if q1:
        #     train_loss = hist.history['loss']
        #     plt.plot(epochX, train_loss, label = 'traning loss')

        plt.xlabel('Epoch', fontsize=15)
        plt.tick_params(axis='x', labelsize=14)
        plt.ylabel('Loss',fontsize=15)
        plt.tick_params(axis='y', labelsize=14)
        plt.title('Loss vs. Epoch',fontsize=15)
        plt.legend(fontsize=10)
        plt.savefig(self.__output1+fileName)


    
    def plot_accuracy(self,hist,fileName=''):
        plt.style.use('seaborn-white')
        test_acc = hist.history['accuracy']
        epochX = list(range(1, len(test_acc) + 1))
        plt.plot(epochX, test_acc, color = 'red', label = 'test accuracy')
        # if q1:
        #     train_acc = hist.history['accuracy']
        #     plt.plot(epochX, train_acc, label = 'training accuracy')  
        plt.xlabel('Epoch',fontsize=15)
        plt.tick_params(axis='x', labelsize=14)
        plt.ylabel('Accuracy',fontsize=15)
        plt.tick_params(axis='y', labelsize=14)
        plt.title('Accuracy vs. Epoch',fontsize=15)  
        plt.legend(loc='lower right',fontsize=10)
        plt.savefig(self.__output1+fileName)