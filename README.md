# Zip Zap Boing

## Table of Contents
1. [Introduction](#introduction)
    1. [Rules](#rules)
2. [Requirements](#requirements)
3. [System](#system)
    1. [Game Environment](#game-environment)
    2. [Wakeup](#game-node)
    3. [Movement](#movement-node)
    4. [Gesture Recognition](#gesture-recognition)
4. [Operation](#operation)


## Introduction

Zip Zap Boing is a very simple game where players gather in a circle and pass a __turn__ around to other players. Given the game's simplicity, we figured that we could teach Shutter how to play, and thus introduce a new means for Shutter to interact with people and its surroundings.

### Rules

Zip Zap Boing is played by _passing_ a __turn__ around a circle. On a player's __turn__, they have a select set of actions that they may choose from.
- If a player wants to pass the turn to the player to their left, they point left and say Zip.
- If they want to choose the player on their right, the point right and say Zap.
- If they want to choose the person that just pointed at them, they say Boing without pointing. (__Note__: This means a player cannot follow a Zip with a Zap, and vice versa.)
- If a player recieves a Boing, they may not choose Boing to redirect the turn. 
- Each time another player is selected in this way, it is their turn to quickly Zip, Zap, or Boing.
- If a player hesitates, doesn’t respond when it is their turn, their word doesn’t match where they are pointing, or they take an improper action, then the game ends and resets.

## Requirements

- [Kinect](https://github.com/Yale-BIM/f22-assignments/blob/master/shutter-notes/kinect.md) (retrieve skeletal pose)
- [Tacotron](https://github.com/yale-img/tacotron2-ros) (text to speech)
    - __Note__: Tacotron uses CUDA 11, while BIM machines currently run CUDA 12. [CUDA 11](https://developer.nvidia.com/cuda-11.0-download-archive) had to be installed as well.
- [Shutter](https://shutter-ros.readthedocs.io/en/latest/index.html) (The robot and it's dependencies)

## System

To play the game of Zip Zap Boing, Shutter has a myriad of nodes that enable it to understand the rules of the game, react to player input, and then output a response.  

The system is composed of a pose interpreter (`interpret_pose.py`), kinect, and rules/game (`zip_zap_boing.py`) nodes. The kinect retrieves skeletal pose data from players, and the interpreter node translates arm positional data into gestures that Shutter can understand. The actual game, and its rules, takes in these gestures and uses them to change the game state and predict moves when it is Shutter's turn.

Shutter must also output its decision, so that other players may follow along and continue the game. These outputs consist of a tacotron text-to-speech node and a motion node (`shutter_react.py`). When Shutter makes a prediction, it publishes a String to a speech node that reads the text through Shutter's (or the attached computer's) speakers. The motion node has a set of 3 movement actions it may take based on the action chosen by Shutter (published as a String).  

1. ### Game Environment

    To understand the rules of the game, Shutter calls the `zip_zap_boing.py` script, which trains Shutter how to play the game with an array of players where each player is assigned an index. After a set amount of epochs to train on, the players are replaced input from the kinect.  