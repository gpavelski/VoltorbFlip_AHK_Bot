# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 07:55:58 2022

@author: Charles
"""
## Importing Libraries
import time
from PIL import ImageGrab
from PIL import Image
from ahk import AHK
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from itertools import combinations_with_replacement
from itertools import permutations
import matplotlib.pyplot as plt
from pickle import load
from VoltorbFlip_Calibration import find_locations_above_threshold

os.environ['KMP_DUPLICATE_LIB_OK']='True'

## Declaration of Functions

def getGameScreenshot(win):
    win.activate()
    x, y, width, height = win.rect
    cap = ImageGrab.grab(bbox =(x, y, x+width, y+height))
    return cap

def cropGameScreenshot(image, BoxCoord):
    image = image.crop(box = (BoxCoord[2],BoxCoord[0],BoxCoord[3],BoxCoord[1]))
    return image

def create_testdata(img):
    # This function is necessary for feeding the TensorFlow model with a valid input
    x = []
    img_array = np.array(img)
    img_array = img_array.astype(np.uint8)
    x.append(img_array)
    x = np.array(x)
    x = np.expand_dims(x, axis=3)
    test_image = x.astype('float32') / 255
    return test_image

def processImage(img,Coord, Mode):
    # Do some image processing in the input picture
    boxed = img.crop(Coord)# crop the image
    boxed_array = np.array(boxed) 
    gray_pic = cv2.cvtColor(boxed_array, cv2.COLOR_BGR2GRAY) #converting the image into grayscale
    r, threshold = cv2.threshold(gray_pic, 125, 255, cv2.THRESH_OTSU) #converting the image into grayscale using the histogram method
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(threshold,kernel,iterations=1)
    pil_image = Image.fromarray(dilated)
    
    if Mode == 0: # Sum of Cards, Number of Bombs, Coins

        Digit = pil_image.resize((28,28))
        
        return Digit
    
    elif Mode == 1: # Square Numbers
        th, im_th = cv2.threshold(dilated, 220, 255, cv2.THRESH_BINARY_INV);
        im_floodfill = im_th.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
         
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 0);
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        
        pil_image = Image.fromarray(im_floodfill_inv)
        
        SquareDigit = pil_image.resize((28,28))
        SquareDigit_cropped = SquareDigit.crop((8,7,22,21))
        SquareDigit = SquareDigit_cropped.resize((28,28))
        
        return SquareDigit
    
    elif Mode == 2: # Level Digit:
        InvertedDigit = cv2.bitwise_not(dilated)
        pil_image = Image.fromarray(InvertedDigit)
        LvlDigit = pil_image.resize((28,28))
        
        return LvlDigit
    
def getGameInfo(img, SumCoordinatesLeft, SumCoordinatesRight, NBombsCoordinates):
    # Related to the Grid Picture Pixels:
    
    InfoArray = np.zeros([10,2],dtype = int)
    
    ## Extracting the Digits from the Figure:
    for i in range(10):
        
        SumLeftDigit = processImage(img, SumCoordinatesLeft[i,:], 0)
        SumRightDigit = processImage(img, SumCoordinatesRight[i,:], 0)
        
        test_image = create_testdata(SumLeftDigit)
        SumLeftDigit = np.argmax(model.predict(test_image, verbose = 0))
        
        test_image = create_testdata(SumRightDigit)
        SumRightDigit = np.argmax(model.predict(test_image, verbose = 0))
        
        InfoArray[i,0] = 10*SumLeftDigit + SumRightDigit
        
        NBombsDigit = processImage(img, NBombsCoordinates[i,:],0)

        test_image = create_testdata(NBombsDigit)
        InfoArray[i,1] = np.argmax(model.predict(test_image, verbose = 0))
    
    return InfoArray

def getSquaresInfo(Grid, SqCoordinates):
    
    Squares = np.zeros([5,5],dtype = int)
    
    ## Creating the Game Matrix
    for i in range(5):
        for j in range(5):
            Sqdigit = processImage(Grid, SqCoordinates[5*i+j,:],1)
            
            test_image = create_testdata(Sqdigit)
            Squares[i,j] = np.argmax(model.predict(test_image, verbose=0))
            if Squares[i,j] == 11: # 11 = Unflipped Card
                Squares[i,j] = 0
                
    return Squares

def getCurrentLvl(img, LvlCoordinates):
    LvlDigit = processImage(img,LvlCoordinates, 2)
    test_image = create_testdata(LvlDigit)
    lvl = np.argmax(model.predict(test_image, verbose = 0))
    return lvl

def getCoins(img, CoinsCoord):
    digits = np.zeros(5,dtype=int)
    for i in range(5):
        processedDigit = processImage(img,CoinsCoord[i,:], 0)
        test_image = create_testdata(processedDigit)
        digits[i] = np.argmax(model.predict(test_image, verbose = 0))
                
    text = ''
    text = text.join([str(digits[i]) for i in range(len(digits))])
    coins = int(text)
    return coins

def clickSquare(win, BoxCoord, SquareIndex, SqCoordinates):
    
    x = BoxCoord[2] + SqCoordinates[SquareIndex,0] + 16
    y = BoxCoord[0] + SqCoordinates[SquareIndex,1] + 16
    
    win.activate()
    time.sleep(0.6)    
    ahk.click(x,y,n=3)
    time.sleep(0.2)
    ahk.click(x,y,n=3)
    time.sleep(0.2)


def possibleCards(nBombs,nonzeroSquares,sumSquares,totalSum):
    Spaces = 5 - nBombs - nonzeroSquares
    if Spaces > 0:
        slots = np.ones(Spaces,dtype=int)
        diff = totalSum - sumSquares - Spaces
        L = list(combinations_with_replacement([0,1,2], Spaces))
        L = [slots + L[i] for i in range(len(L)) if np.sum(L[i])== diff]
        L = [np.append(L[i], np.zeros(nBombs,dtype=int)) for i in range(len(L))]
    else:
        L = []
        L.append(np.zeros(nBombs,dtype=int))
    return L

def testMatrix(M,InfoArray):
    t = 0
    # Check if the sum of the columns is correct
    for i in range(5):
        if np.sum(M[:,i]) == InfoArray[5+i,0] and np.count_nonzero(M[:,i] == 0) == InfoArray[i+5,1]:
            t += 1
        else:
            break
    if t == 5:
        return 1
    else:
        return 0
    
def getSolutions(Squares, InfoArray, RowInd, NewPermList):
    ## Brute-force check of the permutations
    solList = []
    M = Squares.copy()

    p = np.zeros(len(RowInd), dtype = int)
    if len(RowInd) == 0:
        pass
    elif len(RowInd) == 1:

        while p[0] < len(NewPermList[RowInd[0]]):
                for i in range(len(RowInd)): # Row
                    ColInd = [ind for ind in range(5) if Squares[RowInd[i]][ind] == 0]
                    for j in range(len(NewPermList[RowInd[i]][0])): # Column
                        M[RowInd[i],ColInd[j]] = NewPermList[RowInd[i]][p[i]][j]
    
                if testMatrix(M,InfoArray) == 1:
                    solList.append(M.copy())
                p[0] += 1
            
    elif len(RowInd) == 2:
        
        while p[0] < len(NewPermList[RowInd[0]]):
            p[1] = 0
            while p[1] < len(NewPermList[RowInd[1]]):
                for i in range(len(RowInd)): # Row
                    ColInd = [ind for ind in range(5) if Squares[RowInd[i]][ind] == 0]
                    for j in range(len(NewPermList[RowInd[i]][0])): # Column
                        M[RowInd[i],ColInd[j]] = NewPermList[RowInd[i]][p[i]][j]
    
                if testMatrix(M,InfoArray) == 1:
                    solList.append(M.copy())
                p[1] += 1
            p[0] += 1
    
    elif len(RowInd) == 3:
        
        while p[0] < len(NewPermList[RowInd[0]]):
            p[1] = 0
            while p[1] < len(NewPermList[RowInd[1]]):
                p[2] = 0
                while p[2] < len(NewPermList[RowInd[2]]):
                    for i in range(len(RowInd)): # Row
                        ColInd = [ind for ind in range(5) if Squares[RowInd[i]][ind] == 0]
                        for j in range(len(NewPermList[RowInd[i]][0])): # Column
                            M[RowInd[i],ColInd[j]] = NewPermList[RowInd[i]][p[i]][j]
        
                    if testMatrix(M,InfoArray) == 1:
                        solList.append(M.copy())
                    p[2] += 1
                p[1] += 1
            p[0] += 1
    
    elif len(RowInd) == 4:
        
        while p[0] < len(NewPermList[RowInd[0]]):
            p[1] = 0
            while p[1] < len(NewPermList[RowInd[1]]):
                p[2] = 0
                while p[2] < len(NewPermList[RowInd[2]]):
                    p[3] = 0
                    while p[3] < len(NewPermList[RowInd[3]]):
                        for i in range(len(RowInd)): # Row
                            ColInd = [ind for ind in range(5) if Squares[RowInd[i]][ind] == 0]
                            for j in range(len(NewPermList[RowInd[i]][0])): # Column
                                M[RowInd[i],ColInd[j]] = NewPermList[RowInd[i]][p[i]][j]
                                
                        if testMatrix(M,InfoArray) == 1:
                            solList.append(M.copy())
                        p[3] += 1
                    p[2] += 1
                p[1] += 1
            p[0] += 1
    
    elif len(RowInd) == 5:
        
        while p[0] < len(NewPermList[RowInd[0]]):
            p[1] = 0
            while p[1] < len(NewPermList[RowInd[1]]):
                p[2] = 0
                while p[2] < len(NewPermList[RowInd[2]]):
                    p[3] = 0
                    while p[3] < len(NewPermList[RowInd[3]]):
                        p[4] = 0
                        while p[4] < len(NewPermList[RowInd[4]]):
                            for i in range(len(RowInd)): # Row
                                ColInd = [ind for ind in range(5) if Squares[RowInd[i]][ind] == 0]
                                for j in range(len(NewPermList[RowInd[i]][0])): # Column
                                    M[RowInd[i],ColInd[j]] = NewPermList[RowInd[i]][p[i]][j]    
                            if testMatrix(M,InfoArray) == 1:
                                solList.append(M.copy())
                            p[4] += 1
                        p[3] += 1
                    p[2] += 1
                p[1] += 1
            p[0] += 1
    
    return solList

def makeDecision(solList,Squares):
    M = np.zeros([5,5,4],dtype=int)
    # Compare solutions to get the probability of each digit
    if len(solList) >= 1:
        for i in range(len(solList)):
            for j in range(5):
                for k in range(5):
                    if solList[i][j][k] == 0:
                        M[j,k,0] +=1
                    elif solList[i][j][k] == 1:
                        M[j,k,1] +=1
                    elif solList[i][j][k] == 2:
                        M[j,k,2] += 1
                    elif solList[i][j][k] == 3:
                        M[j,k,3] += 1
    
        Pdigit = np.zeros([4,5,5])
        for i in range(4): # Digit
            for j in range(5): # Row
                for k in range(5): # Column
                    Pdigit[i,j,k] = M[j,k,i]/len(solList)
    
        DecisionMatrix = (1-Pdigit[0,:,:])*(Pdigit[2,:,:] + Pdigit[3,:,:])
        DecisionMatrix[Squares > 0] = 0
    else:
        return np.zeros([5,5],dtype=int)
    return DecisionMatrix 

def getPossibleValues(InfoArray, Squares):
    PossibleValues = []
    for i in range(len(InfoArray)):           
        if i < 5:
            if np.count_nonzero(Squares[i,:] == 0) != 0:
                L = possibleCards(InfoArray[i,1],np.count_nonzero(Squares[i,:] != 0),np.sum(Squares[i,:]),InfoArray[i,0])
                PossibleValues.append(L)
            else:
                PossibleValues.append([])
        else:
            if np.count_nonzero(Squares[:,i-5] == 0) != 0:
                L = possibleCards(InfoArray[i,1],np.count_nonzero(Squares[:,i-5] != 0),np.sum(Squares[:,i-5]),InfoArray[i,0])
                PossibleValues.append(L)
            else:
                PossibleValues.append([])
    return PossibleValues

def getListofPermutations(PossibleValues):
## List the possible permutations in each line column
    PermList = []            
    for i in range(5):
        if len(PossibleValues[i]) > 0:
            L = []
            for j in range(len(PossibleValues[i])):
                L += [list(set(list(permutations(PossibleValues[i][j]))))]
            PermList.append(L)
        else:
            PermList.append([])
        
        if len(PermList[i]) > 1:
            L = []
            for j in range(len(PermList[i])): 
                L += PermList[i][j]
            PermList[i] = []
            PermList[i].append(L)
    return PermList

def filterPermutations(PermList, Squares, PossibleValues):
## Remove Impossible Permutations     
    NewPermList = [[],[],[],[],[]]
    for i in range(len(PermList)):
        ColInd = [ind for ind in range(5) if Squares[i][ind] == 0]
        if len(PermList[i]) > 0:
            for j in range(len(PermList[i][0])): # Analyse each permutation
                flag = 0
                for k in range(len(PermList[i][0][0])): # Column index
                    if PermList[i][0][j][k] not in np.unique(PossibleValues[5+ColInd[k]][:]):
                        flag = 1
                if flag == 0:
                    NewPermList[i].append(PermList[i][0][j])
    return NewPermList

def simpleProbTest(InfoArray, Squares):
    ## Probability of opening each card
    PossibleValues = getPossibleValues(InfoArray, Squares)
    ProbMatrix = np.zeros([4,5,5])
    for digit in range(4):
        for i in range(5):
            for j in range(5):
                if Squares[i,j] == 0:
                    ProbRow = 0
                    ProbCol = 0
                    for k in range(len(PossibleValues[i])):
                        ProbRow += (1/len(PossibleValues[i]))*np.count_nonzero(PossibleValues[i][k] == digit)/len(PossibleValues[i][k])
                    for k in range(len(PossibleValues[5+j])):
                        ProbCol += (1/len(PossibleValues[5+j]))*np.count_nonzero(PossibleValues[5+j][k] == digit)/len(PossibleValues[5+j][k])
                    ProbMatrix[digit,i,j] = ProbRow*ProbCol
    DecisionMatrix = (1-ProbMatrix[0,:,:])*(ProbMatrix[2,:,:] + ProbMatrix[3,:,:])
    return DecisionMatrix

def isVoltorbFlipOpen(img, template):
    sorted_locations = find_locations_above_threshold(img, template)
    return bool(len(sorted_locations))
        
def LoadData(cur_dir):

    file = open(cur_dir + '\\Data\\BoxCoord.p', 'rb')
    BoxCoord = load(file)
    file.close()
    
    file = open(cur_dir + '\\Data\\SqCoordinates.p', 'rb')
    SqCoordinates = load(file)
    file.close()
    
    file = open(cur_dir + '\\Data\\SumCoordinatesLeft.p', 'rb')
    SumCoordinatesLeft = load(file)
    file.close()
    
    file = open(cur_dir + '\\Data\\SumCoordinatesRight.p', 'rb')
    SumCoordinatesRight = load(file)
    file.close()
    
    file = open(cur_dir + '\\Data\\NBombsCoordinates.p', 'rb')
    NBombsCoordinates = load(file)
    file.close()
    
    file = open(cur_dir + '\\Data\\CoinsCoordinates.p', 'rb')
    CoinsCoordinates = load(file)
    file.close()
    
    file = open(cur_dir + '\\Data\\LvlCoordinates.p', 'rb')
    LvlCoordinates = load(file)
    file.close()
    
    return BoxCoord, SqCoordinates, SumCoordinatesLeft, SumCoordinatesRight, NBombsCoordinates, CoinsCoordinates, LvlCoordinates 

if __name__ == "__main__":
    
    ## Importing Files and Models
    cur_dir = os.getcwd()
    savepath = cur_dir + '\\Digit.png'
    
    model=load_model(cur_dir + '\\Models\\DigitsModel.h5')
    savedModelWeights = model.load_weights(cur_dir + '\\Models\\DigitsModelWeights.h5')
    
    model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    ## Variable Declarations
    TargetCoins = 700
    
    ## Getting a screenshot from the MelonDS screen
    ahk = AHK()
    
    win = ahk.find_window(title=b'melonDS 0.9.5') # Find the opened window
     
    win.set_position((0, 0))
    win.width = 581
    win.height = 710
    
    # Import the data
    BoxCoord, SqCoordinates, SumCoordinatesLeft, SumCoordinatesRight, NBombsCoordinates, CoinsCoordinates, LvlCoordinates = LoadData(cur_dir)
    
    img = getGameScreenshot(win)
    Grid = cropGameScreenshot(img, BoxCoord)
    
    template_path = cur_dir + "\\Patterns\\patternLvl.png" 
    template = Image.open(template_path)
    
    GameOpen = isVoltorbFlipOpen(img, template)
    
    if GameOpen:
        coins = getCoins(img, CoinsCoordinates)
        it = 0
        stats = []
        while coins < TargetCoins:
        
            # Get the initial information of the grid 
            img = getGameScreenshot(win) # Get the screenshot
            Grid = cropGameScreenshot(img, BoxCoord)
            InfoArray = getGameInfo(img, SumCoordinatesLeft, SumCoordinatesRight, NBombsCoordinates) # Extract information
            Squares = getSquaresInfo(Grid,SqCoordinates) # Extract the current grid
        
            while len(Squares[Squares > 3]) > 0:
            # If there is no card to be flipped, then there is no mean in clicking squares.
        
                win.activate() # Initial click to activate the next game if necessary
                x,y,width,height = win.rect # Get the coordinates in the screen where the game is open
                ahk.click((x+403,512+y),n=3)
                time.sleep(0.1)
                ahk.click((x+403,512+y),n=3) # Just to make sure the click happened
                time.sleep(1)
        
                img = getGameScreenshot(win) # Get the screenshot
                Grid = cropGameScreenshot(img, BoxCoord)
                InfoArray = getGameInfo(img, SumCoordinatesLeft, SumCoordinatesRight, NBombsCoordinates) # Extract information
                Squares = getSquaresInfo(Grid,SqCoordinates) # Extract the current grid
        
            lvl = getCurrentLvl(img, LvlCoordinates) # Read the current lvl of the game
            coins = getCoins(img, CoinsCoordinates) # Read the current number of coins 
            print("iteration", it, ", Lvl ", lvl, ", Coins ", coins) # Just for tracking progress
            stats.append([it, lvl, coins])
            it += 1
            
            for i in range(len(InfoArray)):
                if InfoArray[i,1] == 0:     ## Click rows/columns with 0 Bombs:   
                    if i < 5:
                        while np.sum(Squares[i,:]) < InfoArray[i,0]:
                            for j in range(5):
                                if Squares[i,j] == 0:
                                    clickSquare(win, BoxCoord, 5*i+j, SqCoordinates)
                            img = getGameScreenshot(win)
                            Grid = cropGameScreenshot(img, BoxCoord)
                            Squares = getSquaresInfo(Grid,SqCoordinates)
                    else:
                        while np.sum(Squares[:,i-5]) < InfoArray[i,0]:
                            for j in range(5):
                                if Squares[j,i-5] == 0:
                                    clickSquare(win, BoxCoord, 5*j+(i-5), SqCoordinates)
                            img = getGameScreenshot(win)
                            Grid = cropGameScreenshot(img, BoxCoord)
                            Squares = getSquaresInfo(Grid,SqCoordinates)
        
            ## Trying to solve the problem
            while 10 not in Squares or 0 in Squares:
        
                ## Update of the Matrix:
                img = getGameScreenshot(win)
                Grid = cropGameScreenshot(img, BoxCoord)
                Squares = getSquaresInfo(Grid,SqCoordinates)
        
                if 10 not in Squares:
        
                    img = getGameScreenshot(win)
                    Grid = cropGameScreenshot(img, BoxCoord)
                    Squares = getSquaresInfo(Grid,SqCoordinates)
            
                    PossibleValues = getPossibleValues(InfoArray, Squares)
                    PermList = getListofPermutations(PossibleValues)
                    NewPermList = filterPermutations(PermList, Squares, PossibleValues)
                                                     
                    RowInd = [i for i in range(5) if np.sum(Squares[i,:]) < InfoArray[i,0]]
            
                    NumPerms = np.prod([len(NewPermList[i]) for i in range(5) if len(NewPermList[i]) > 0])
            
                    if NumPerms <= 120000:
                        solList = getSolutions(Squares, InfoArray, RowInd, NewPermList)
                        DecisionMatrix = makeDecision(solList,Squares)
                    else:
                        DecisionMatrix = simpleProbTest(InfoArray, Squares)
            
                    ## Click on a likely square
                    ChosenSquare = np.argmax(DecisionMatrix)
                    clickSquare(win, BoxCoord, ChosenSquare, SqCoordinates)
            
            coins = getCoins(img, CoinsCoordinates) # Update the number of coins
        
        ## Plotting the results
        coins = getCoins(img, CoinsCoordinates) # Update the number of coins
        lvl = getCurrentLvl(img, LvlCoordinates) # Read the current lvl of the game
        stats.append([it, lvl, coins])
        
        stats = np.array(stats, dtype = int)

        plt.figure(0)
        plt.plot(range(len(stats)),stats[:,1])
        plt.xlabel('Iteration')
        plt.ylabel('Level')
        plt.figure(1)
        plt.plot(range(len(stats)),stats[:,2])
        plt.xlabel('Iteration')
        plt.ylabel('Number of Coins')
        
    else:
        print("Voltorb Flip is not open. Make sure to start the game")