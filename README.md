# VoltorbFlip_AHK_Bot
Uses probability for playing the VoltorbFlip mini-game from Pokemon Heart Gold.

The idea is to use AutoHotKey (AHK) for playing the Voltorb Flip mini-game in the Game Corner of Pokémon HeartGold/SoulSilver.
For this, a Neural Network has been trained using Keras (Tensorflow) for recognizing the digits that appear on the screen.
For obtaining the screenshots, I use Pillow ImageGrab function.
For automatically defining the pixel locations, I use CV2 correlation method for comparing the game screenshot to a defined pattern, and based on the best correlation, it returns the pixels locations.
It has been tested using MelonDS version 0.9.5.
The algorithm for solving it is based on comparing the possible solutions and decide what is the most probable output. However, in the start of each episode, the number of Possible Solutions is too large. Thus, we use a basic method first and when the number of solutions decrease, use the improved method.

VoltorbFlip is based on luck and skills, so there is no perfect solution. Using this bot it is possible to set up a target number of coins, let the bot play until the target value is achieved.
