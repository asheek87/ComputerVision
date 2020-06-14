import cv2
import numpy as np
class Image():
    '''
    This class is the data structure for storing image info 
    and trsanformation processes of an image
    '''
    
    def __init__(self,imgFile,imgLblStr,imgLblInt,imgGrp,fileName):
        self.__origImg=imgFile
        self.__strLable=imgLblStr
        self.__intLable=imgLblInt
        self.fileName=fileName
        self.imgGrp=imgGrp # Train or test image
    
    def transform(self,aSide):
        '''
        Image will be resize and changed color
        Return a list of an images which is resized
        '''
        #resize original image to square, change to RGB channel and normalize to array
        self.__imageResize = cv2.resize(self.__origImg, (aSide, aSide))
        self.__imageResize = cv2.cvtColor(self.__imageResize,cv2.COLOR_BGR2RGB)
        imglist=[self.__imageResize]

        return imglist

    
    def getIntLable(self):
        '''
        return int lable of image
        '''
        return self.__intLable

    def getStrLable(self):
        '''
        return string lable of image
        '''
        return self.__strLable

    def getGrpType(self):
        return self.imgGrp
    
    def getFileName(self):
        return self.fileName
