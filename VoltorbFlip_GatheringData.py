# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:26:53 2023

@author: Guilherme
"""

## Importing Libraries
import time
from PIL import ImageGrab
from PIL import Image
from ahk import AHK
from pickle import load, dump
import os
import numpy as np
import cv2

def LoadData(cur_dir):
    # These coordinates represent the four corners of the bottom window.
    # Let it untouched
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

def getGameScreenshot(win):
    win.activate()
    x, y, width, height = win.rect
    cap = ImageGrab.grab(bbox =(x, y, x+width, y+height))
    return cap

def cropGameScreenshot(image, BoxCoord):
    image = image.crop(box = (BoxCoord[2],BoxCoord[0],BoxCoord[3],BoxCoord[1]))
    return image

def processImage(img,Coord, Mode):
    # Do some image processing in the input picture
    boxed = img.crop(Coord)# crop the image
    boxed_array = np.array(boxed) 
    gray_pic = cv2.cvtColor(boxed_array, cv2.COLOR_BGR2GRAY) #converting the image into grayscale
    r, threshold = cv2.threshold(gray_pic, 125, 255, cv2.THRESH_OTSU) #converting the image into grayscale using the histogram method
    #color_converted = cv2.cvtColor(threshold, cv2.COLOR_BGR2RGB)
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

def update_control_number(filename):
    file = open(filename, 'rb')
    control_number = load(file)
    file.close()
    
    control_number += 1
    
    file = open(filename, 'wb')
    dump(control_number, file)
    file.close()
    
    return control_number - 1

if __name__ == "__main__":
    
    cur_dir = os.getcwd()
    ahk = AHK()
    win = ahk.find_window(title=b'melonDS 0.9.5')

    win.set_position((0, 0))
    win.width = 581
    win.height = 710

    # Import the data
    BoxCoord, SqCoordinates, SumCoordinatesLeft, SumCoordinatesRight, NBombsCoordinates, CoinsCoordinates, LvlCoordinates = LoadData(cur_dir)

    # Get a screenshot from the game window
    gameScreen = getGameScreenshot(win)
    squares_image = cropGameScreenshot(gameScreen, BoxCoord)
    
    control_number = update_control_number('Data\\control_number.txt')
    digits_folder = cur_dir + "\\Digits\\"
    # Square Digits
    for i in range(5):
        for j in range(5):
            savepath = digits_folder + str(control_number) + "_" + str(5*i+j) + ".png"
            Sqdigit = processImage(squares_image, SqCoordinates[5*i+j,:],1)
            Sqdigit.save(savepath)
    
    # Sum of Cards Digits:
    for i in range(10):
        savepathL = digits_folder + str(control_number) + "_L" + str(i) + ".png"
        savepathR = digits_folder + str(control_number) + "_R" + str(i) + ".png"
        SumLeftDigit = processImage(gameScreen, SumCoordinatesLeft[i,:], 0)
        SumRightDigit = processImage(gameScreen, SumCoordinatesRight[i,:], 0)
        SumLeftDigit.save(savepathL)
        SumRightDigit.save(savepathR)
        
    # Number of Bombs:
    for i in range(10):
        savepath = digits_folder + str(control_number) + "_B" + str(i) + ".png"
        NBombsDigit = processImage(gameScreen, NBombsCoordinates[i,:], 0)
        NBombsDigit.save(savepath)
    
    # Number of Coins:
    for i in range(5):
        savepath = digits_folder + str(control_number) + "_C" + str(i) + ".png"
        NBombsDigit = processImage(gameScreen, CoinsCoordinates[i,:], 0)
        NBombsDigit.save(savepath)
        
    # Level
    savepath = digits_folder + str(control_number) + "_Lvl0.png"
    LevelDigit = processImage(gameScreen, LvlCoordinates, 2)
    LevelDigit.save(savepath)
