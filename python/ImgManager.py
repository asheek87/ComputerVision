import os
import cv2
import numpy as np
import pandas as pd
import shutil


from Image import Image
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing


class ImgManager():
    '''
    This class is reposnible to handle all image processing steps
    '''
    __RootFolder='birds_dataset'
    # __labelFileCsv='test_trainLabels_bird.csv'
    __fileNameCols='filename'
    __labelCols='label'
    __labelEncodeCols='label_LE'
    __labelCount="count"
    __dataFolder='birds'
    __trainFolder='TrainImg'
    __testFolder='TestImg'

    __slash='//'
    
    __Test='Test'
    __Train='Train'
    __Validate='Validate'
    __valPrefix='val_'
    

    def __init__(self):
        '''
        test size: between 0.1 to 0.9. Up to 1 decimal place Default is 0.1
        '''
        self.__myImgList=[]
        self.__myProcessImgTrainList=[]
        self.__myProcessImgIntLableTrainList=[]

        self.__myProcessImgTestList=[]
        self.__myProcessImgIntLableTestList=[]

        self.__dictLable={}
        self.__approveDict={}
     
        self.__selectedDim=224
        self.__createDir(ImgManager.__trainFolder)
        self.__createDir(ImgManager.__testFolder)

    def __createDir(self,outputFolder):
        if os.path.exists(outputFolder):
            shutil.rmtree(outputFolder)
            os.makedirs(outputFolder,exist_ok=True)
        else:
            os.makedirs(outputFolder,exist_ok=True)

    def analyseLabel(self,fileName):
        csvFile=fileName
        df = pd.read_csv(ImgManager.__RootFolder+ImgManager.__slash+csvFile)
        # convert original dataframe to Dictionary. key is file name, value is the label as a list
        self.__dictLable=df.set_index(ImgManager.__fileNameCols).T.to_dict('list')

        #count labels and add new column
        df[ImgManager.__labelCount] = df.groupby(ImgManager.__labelCols)[ImgManager.__labelCols].transform(ImgManager.__labelCount)
        #drop duplicate rows first then drop file name column
        df.drop(ImgManager.__fileNameCols, axis = 1,inplace=True)
        df.drop_duplicates(keep='last',inplace=True)
        df.sort_values(ImgManager.__labelCols, inplace=True)
        df.reset_index(drop = True, inplace = True)
        self.__uniqueVal=df[ImgManager.__labelCols].nunique()
        print('Original number of unique labels: '+ImgManager.__labelCols+' :'+str(self.__uniqueVal))
        
        return df

    def filterApprovedBirds(self,dataframe,threshold):
        '''
        Find labels which is more or equal to  number defined theshold.
        ie. get all birds which has a count more than or equal to the thereshold
        '''
        self.__maxNumPerClassImg=threshold
        df=dataframe.copy()
        indexNames = df[ df[ImgManager.__labelCount] < threshold ].index
        # Delete these row indexes from dataFrame
        df.drop(indexNames , inplace=True)
        # perform label encoding on the 'label'column to get a numeric value for lable
        le=preprocessing.LabelEncoder()
        df[ImgManager.__labelEncodeCols]=le.fit_transform(df[ImgManager.__labelCols])
        df.reset_index(drop = True, inplace = True)
        self.__uniqueVal=df[ImgManager.__labelCols].nunique()
        print('Number of unique labels to be processed: '+str(self.__uniqueVal))
        total = df[ImgManager.__labelCount].sum()
        print('Total number of images to be processed: '+str(total))
        #create a dictionary of approved labels. key is label . value is a list of  count and label encoded value
        self.__approveDict=df.set_index(ImgManager.__labelCols).T.to_dict('list')
        
        return df

    def saveLabelToTxt(self,outFolder):
        labelList=list(self.__approveDict.keys())
        with open(outFolder+'Labels.txt', 'w') as filehandle:
            for alabel in labelList:
                filehandle.write('%s\n' % alabel)

    def __getFileName(self,file):
        names=file.split('.')
        return names[0]

    def getNoOfUniqueOutputs(self):
        return self.__uniqueVal


    def readImages(self):
        '''
        Read all images.For each class image, read up to 10 images. To ensure a balanced dataset
        Return None
        '''
        labelsDict=self.__dictLable
        testlist=[]
        counter=0
        trainimg=0
        listofImages=[]
        

        print('Reading images from folder '+ImgManager.__dataFolder+' ...')
        for root, dirs, files in os.walk(ImgManager.__RootFolder + ImgManager.__slash + ImgManager.__dataFolder):
            for aFile in files:
                fileName=self.__getFileName(aFile)
                label=labelsDict.get(fileName)[0]
                #save img,string and int label to be save into img obj
                if label in list(self.__approveDict.keys()):
                    strLabel=label
                    if listofImages.count(str(strLabel))<=self.__maxNumPerClassImg:
                        listofImages.append(strLabel)
                        #get the integer label
                        intLabel=self.__approveDict.get(label)[1]
                        img=cv2.imread(ImgManager.__RootFolder+ ImgManager.__slash + ImgManager.__dataFolder + ImgManager.__slash + aFile)
                        # make one image of a class to be test image
                        if label not in testlist:
                            myImg=Image(img,strLabel,intLabel,ImgManager.__Test,fileName)
                            testlist.append(label)
                        else:
                            myImg=Image(img,strLabel,intLabel,ImgManager.__Train,fileName)
                            trainimg+=1
                        
                        self.__myImgList.append(myImg)
                        counter+=1

        print('Smallest side: '+str(self.__selectedDim))
        print('Number of overall images: '+str(counter))
        print('Number of test images: '+str(len(testlist)))
        print('Number of train images: '+str(trainimg))
        print('Reading images from folder DONE \n')

        
    def getStrKeyFromVal(self,val):
        for key, values in self.__approveDict.items(): 
            if val == values[1]: 
                aKey= key
                break
            else: 
                aKey=None 
        return aKey
        
    def getSideDimension(self):
        return self.__selectedDim
    
    def procesImages(self):
        '''
        Will take in original images and for each image will generate new images
        after undergoing transformation. 
        Return None
        '''
        print('Processing all images... ')
        
        for myImage in self.__myImgList:
            transformList=myImage.transform(self.__selectedDim)
            for aTransformImg in transformList:
                if(myImage.getGrpType()==ImgManager.__Train):
                    self.__myProcessImgTrainList.append(aTransformImg)
                    self.__myProcessImgIntLableTrainList.append(myImage.getIntLable())
                    img = cv2.cvtColor(aTransformImg, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(ImgManager.__trainFolder+ImgManager.__slash+myImage.getStrLable()+"_"+myImage.getFileName()+'.jpg', img)

                elif(myImage.getGrpType()==ImgManager.__Test):
                    self.__myProcessImgTestList.append(aTransformImg)
                    self.__myProcessImgIntLableTestList.append(myImage.getIntLable())
                    img = cv2.cvtColor(aTransformImg, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(ImgManager.__testFolder+ImgManager.__slash+myImage.getStrLable()+"_"+myImage.getFileName()+'.jpg', img)

        print('Test Images '+str(len(self.__myProcessImgTestList)))
        print('Train Images '+str(len(self.__myProcessImgTrainList)))
        print('Processing all images DONE ')
    
    #TRAIN
    def __getProcessedImagesTrainList(self):
        # convert to array and normalize the RGB values
        return np.array(self.__myProcessImgTrainList,dtype="float32")/255
        

    def __getIntlablesTrainList(self):
        return to_categorical(np.array(self.__myProcessImgIntLableTrainList))
    
    #TEST
    def __getProcessedImagesTestList(self):
        # convert to array and normalize the RGB values
        return np.array(self.__myProcessImgTestList,dtype="float32")/255
        
    
    def __getIntlablesTestList(self):
        #one-hot encoding on the labels
        return to_categorical(np.array(self.__myProcessImgIntLableTestList))


    def get_Train_Test_Data(self):
        return self.__getProcessedImagesTrainList(),self.__getIntlablesTrainList(),self.__getProcessedImagesTestList(),self.__getIntlablesTestList()






