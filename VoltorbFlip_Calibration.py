# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:55:14 2023

@author: Guilherme
"""

import cv2
import numpy as np
from PIL import Image
from PIL import ImageGrab
import pickle
import os
from ahk import AHK
import time

def find_locations_above_threshold(image, template, threshold=0.8):
    # Convert the Pillow image to a NumPy array
    rgb_img = image.convert('RGB')
    image_np = np.array(rgb_img)
    rgb_temp = template.convert('RGB')
    template_np = np.array(rgb_temp)
    
    # Match the template in the image
    result = cv2.matchTemplate(image_np, template_np, cv2.TM_CCOEFF_NORMED)

    # Find the locations where correlation is above the threshold
    locs = np.where(result >= threshold)

    # Transpose the locs array to get a list of (y, x) coordinates
    locations = list(zip(*locs[::-1]))

    # Sort the locations by the first coordinate (y-coordinate)
    sorted_locations = sorted(locations, key=lambda x: x[0])

    return sorted_locations

def find_highest_correlation(image, template):
    # Convert the Pillow image to a NumPy array
    rgb_img = image.convert('RGB')
    image_np = np.array(rgb_img)
    rgb_temp = template.convert('RGB')
    template_np = np.array(rgb_temp)

    # Match the template in the image
    result = cv2.matchTemplate(image_np, template_np, cv2.TM_CCOEFF_NORMED)

    # Find the coordinates of the highest correlation
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    return max_loc

def getGridInfo(x,y,width,height):
    cap = ImageGrab.grab(bbox =(x, y, x+width, y+height))
    gridPic = np.array(cap)
    return gridPic

def getGameScreenshot(win):
    win.activate()
    x,y,width,height = win.rect
    
    gridPic = getGridInfo(x,y,width,height)
    img = Image.fromarray(gridPic)

    return img

if __name__ == "__main__":
    # Load the image and template (replace these paths with your own images)
    cur_dir = os.getcwd()
    folder_path = cur_dir + '\\Patterns\\'
    # image_path = folder_path + "img.png"
    template_path = folder_path + "pattern.png"
    
    ## Getting a screenshot from the MelonDS screen
    ahk = AHK()
    
    win = ahk.find_window(title=b'melonDS 0.9.5') # Find the opened window
    
    win.set_position((0, 0))
    win.width = 581
    win.height = 710
    
    image = getGameScreenshot(win) # Get the screenshot
    # image = Image.open(image_path)
    template = Image.open(template_path)
    
    # Find locations where correlation is above the threshold (e.g., 0.8) and sort them by the first coordinate
    threshold = 0.85
    sorted_locations = find_locations_above_threshold(image, template, threshold)
    
    first_square = sorted_locations[0]
    last_square = sorted_locations[-1] # Remember to do a check first
    pattern_length = len(np.array(template))
    inter_blocks_len = sorted_locations[1][1]-sorted_locations[0][1]
    
    BoxCoord = np.array([first_square[1], last_square[1] + pattern_length, 
                         first_square[0], last_square[0] + pattern_length]) # Top, Bottom, Left, Right corners
    
    # Left, Up, Right, Bottom
    cropped_box = image.crop(box = (BoxCoord[2], BoxCoord[0], BoxCoord[3], BoxCoord[1]))
    
    Top, Left = 0,0
    
    # These coordinates represent the pixels where the numbers containing
    # the cards are located. Used for extracting the digits for the Computer vision 
    # step. The squares are numbered 0 to 24. 
    SqCoordinates = np.zeros([25,4],dtype = int)
    # Left, Up, Right, Bottom
    SqCoordinates[0,:] = Left                     , Top                   , Left+pattern_length                     , Top+pattern_length
    SqCoordinates[1,:] = Left+1*inter_blocks_len  , Top                   , Left+pattern_length+1*inter_blocks_len  , Top+pattern_length
    SqCoordinates[2,:] = Left+2*inter_blocks_len  , Top                   , Left+pattern_length+2*inter_blocks_len  , Top+pattern_length
    SqCoordinates[3,:] = Left+3*inter_blocks_len  , Top                   , Left+pattern_length+3*inter_blocks_len  , Top+pattern_length
    SqCoordinates[4,:] = Left+4*inter_blocks_len  , Top                   , Left+pattern_length+4*inter_blocks_len  , Top+pattern_length
    
    SqCoordinates[5,:] = Left                     , Top+1*inter_blocks_len, Left+pattern_length                     , Top+pattern_length+1*inter_blocks_len
    SqCoordinates[6,:] = Left+1*inter_blocks_len  , Top+1*inter_blocks_len, Left+pattern_length+1*inter_blocks_len  , Top+pattern_length+1*inter_blocks_len
    SqCoordinates[7,:] = Left+2*inter_blocks_len  , Top+1*inter_blocks_len, Left+pattern_length+2*inter_blocks_len  , Top+pattern_length+1*inter_blocks_len
    SqCoordinates[8,:] = Left+3*inter_blocks_len  , Top+1*inter_blocks_len, Left+pattern_length+3*inter_blocks_len  , Top+pattern_length+1*inter_blocks_len
    SqCoordinates[9,:] = Left+4*inter_blocks_len  , Top+1*inter_blocks_len, Left+pattern_length+4*inter_blocks_len  , Top+pattern_length+1*inter_blocks_len
    
    SqCoordinates[10,:] = Left                    , Top+2*inter_blocks_len, Left+pattern_length                     , Top+pattern_length+2*inter_blocks_len
    SqCoordinates[11,:] = Left+1*inter_blocks_len , Top+2*inter_blocks_len, Left+pattern_length+1*inter_blocks_len  , Top+pattern_length+2*inter_blocks_len
    SqCoordinates[12,:] = Left+2*inter_blocks_len , Top+2*inter_blocks_len, Left+pattern_length+2*inter_blocks_len  , Top+pattern_length+2*inter_blocks_len
    SqCoordinates[13,:] = Left+3*inter_blocks_len , Top+2*inter_blocks_len, Left+pattern_length+3*inter_blocks_len  , Top+pattern_length+2*inter_blocks_len
    SqCoordinates[14,:] = Left+4*inter_blocks_len , Top+2*inter_blocks_len, Left+pattern_length+4*inter_blocks_len  , Top+pattern_length+2*inter_blocks_len
    
    SqCoordinates[15,:] = Left                    , Top+3*inter_blocks_len, Left+pattern_length                     , Top+pattern_length+3*inter_blocks_len
    SqCoordinates[16,:] = Left+1*inter_blocks_len , Top+3*inter_blocks_len, Left+pattern_length+1*inter_blocks_len  , Top+pattern_length+3*inter_blocks_len
    SqCoordinates[17,:] = Left+2*inter_blocks_len , Top+3*inter_blocks_len, Left+pattern_length+2*inter_blocks_len  , Top+pattern_length+3*inter_blocks_len
    SqCoordinates[18,:] = Left+3*inter_blocks_len , Top+3*inter_blocks_len, Left+pattern_length+3*inter_blocks_len  , Top+pattern_length+3*inter_blocks_len
    SqCoordinates[19,:] = Left+4*inter_blocks_len , Top+3*inter_blocks_len, Left+pattern_length+4*inter_blocks_len  , Top+pattern_length+3*inter_blocks_len
    
    SqCoordinates[20,:] = Left                    , Top+4*inter_blocks_len, Left+pattern_length                     , Top+pattern_length+4*inter_blocks_len
    SqCoordinates[21,:] = Left+1*inter_blocks_len , Top+4*inter_blocks_len, Left+pattern_length+1*inter_blocks_len  , Top+pattern_length+4*inter_blocks_len
    SqCoordinates[22,:] = Left+2*inter_blocks_len , Top+4*inter_blocks_len, Left+pattern_length+2*inter_blocks_len  , Top+pattern_length+4*inter_blocks_len
    SqCoordinates[23,:] = Left+3*inter_blocks_len , Top+4*inter_blocks_len, Left+pattern_length+3*inter_blocks_len  , Top+pattern_length+4*inter_blocks_len
    SqCoordinates[24,:] = Left+4*inter_blocks_len , Top+4*inter_blocks_len, Left+pattern_length+4*inter_blocks_len  , Top+pattern_length+4*inter_blocks_len
    
    pattern_names = ['patternR1.png', 'patternR2.png', 'patternR3.png', 'patternR4.png', 'patternR5.png',
                     'patternC1.png', 'patternC2.png', 'patternC3.png', 'patternC4.png', 'patternC5.png']
    
    pattern_rows = 'patternRows.png'
    pattern_cols = 'patternCols.png'
    rows_pattern_path = folder_path + pattern_rows
    rows_pattern = Image.open(rows_pattern_path)
    location_rows = find_highest_correlation(image, rows_pattern)
    cols_pattern_path = folder_path + pattern_cols
    cols_pattern = Image.open(cols_pattern_path)
    location_cols = find_highest_correlation(image, cols_pattern)
    
    # These coordinates represent the pixels where the numbers containing
    # the sum of Voltorbs are located. Used for extracting the digits for the
    # Computer vision step
    SumCoordinatesLeftDigit = np.zeros([10,4],dtype=int)
    SumCoordinatesRightDigit = np.zeros([10,4],dtype=int)
    for i in range(5):
        pattern_path = folder_path + pattern_names[i]
        pattern = Image.open(pattern_path)
        location = find_highest_correlation(rows_pattern, pattern)
        pattern_size = np.size(np.array(pattern),1)
        Left, Top = location_rows[0] + location[0] + pattern_size, location_rows[1] + location[1]
        SumCoordinatesLeftDigit[i,:] = Left, Top, Left + pattern_size, Top + pattern_size
        SumCoordinatesRightDigit[i,:] = Left + pattern_size, Top, Left + 2*pattern_size, Top + pattern_size
    
    for i in range(5,10):
        pattern_path = folder_path + pattern_names[i]
        pattern = Image.open(pattern_path)
        location = find_highest_correlation(cols_pattern, pattern)
        pattern_size = np.size(np.array(pattern),1)
        Left, Top = location_cols[0] + location[0] + pattern_size, location_cols[1] + location[1]
        SumCoordinatesLeftDigit[i,:] = Left, Top, Left + pattern_size, Top + pattern_size
        SumCoordinatesRightDigit[i,:] = Left + pattern_size, Top, Left + 2*pattern_size, Top + pattern_size
    
    # These coordinates represent the pixels where the numbers containing
    # the number of Voltorbs are located. Used for extracting the digits for the
    # Computer vision step
    NBombsCoordinates = np.zeros([10,4],dtype=int)
    for i in range(10):
        NBombsCoordinates[i,:] = SumCoordinatesRightDigit[i,0], SumCoordinatesRightDigit[i,3] + 8, SumCoordinatesRightDigit[i,2], SumCoordinatesRightDigit[i,3] + 21 
    
    ## Get the number of coins 5 digit coordinates
    coins_path = folder_path + 'PatternCoins.png'
    coins_template = Image.open(coins_path)
    location = find_highest_correlation(image, coins_template)
    Left = location[0] + np.size(np.array(coins_template), 1)
    Top = location[1]
    
    CoinsCoordinates = np.zeros([5,4], dtype = int)
    for i in range(5):
        CoinsCoordinates[i,:] = Left + i*26, Top, Left + (i+1)*26, Top + 46
    
    ## Get the current Level 1 digit coordinates
    Lvl_path = folder_path + 'patternLvl.png'
    Lvl_template = Image.open(Lvl_path)
    location = find_highest_correlation(image, Lvl_template)
    Left = location[0] + np.size(np.array(Lvl_template), 1)
    Top = location[1]
    
    LvlCoordinates = np.array([Left, Top, Left + 13, Top + 20], dtype = int)
    
    data_dir = cur_dir + '\\Data\\'
    
    ## Saving values to files
    file = open(data_dir + 'BoxCoord.p', 'wb')
    pickle.dump(BoxCoord, file)
    file.close()
    
    file = open(data_dir + 'SqCoordinates.p', 'wb')
    pickle.dump(SqCoordinates, file)
    file.close()
    
    file = open(data_dir + 'SumCoordinatesLeft.p', 'wb')
    pickle.dump(SumCoordinatesLeftDigit, file)
    file.close()
    
    file = open(data_dir + 'SumCoordinatesRight.p', 'wb')
    pickle.dump(SumCoordinatesRightDigit, file)
    file.close()
    
    file = open(data_dir + 'NBombsCoordinates.p', 'wb')
    pickle.dump(NBombsCoordinates, file)
    file.close()
    
    file = open(data_dir + 'CoinsCoordinates.p', 'wb')
    pickle.dump(CoinsCoordinates, file)
    file.close()
    
    file = open(data_dir + 'LvlCoordinates.p', 'wb')
    pickle.dump(LvlCoordinates, file)
    file.close()