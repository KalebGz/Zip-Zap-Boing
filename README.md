# Zip Zap Boing

## Table of Contents
1. [Introduction](#introduction)
    1. [Rules](#rules)
2. [Requirements](#requirements)
3. [System](#system)
    1. [Game Environment](#game-environment)
    2. [Wakeup & Movement](#wakeup-&-movement)
    4. [Gesture Recognition](#gesture-recognition)
4. [Operation](#operation)
5. [Video Demo](#demo)

## Introduction

Zip Zap Boing is a very simple game where players gather in a circle and pass a __turn__ around to other players. Given the game's simplicity, we figured that we could teach Shutter how to play, and thus introduce a new means for Shutter to interact with people and its surroundings.

### Rules

Zip Zap Boing is played by _passing_ a __turn__ around a circle. On a player's __turn__, they have a select set of actions that they may choose from.
- If a player wants to pass the turn to the player to their left, they point left and say Zip.
- If they want to choose the player on their right, point right and say Zap.
- If they want to choose the person that just pointed at them, they say Boing without pointing. (__Note__: This means a player cannot follow a Zip with a Zap, and vice versa.)
- If a player receives a Boing, they may not choose Boing to redirect the turn. 
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

    To understand the rules of the game, Shutter calls the `zip_zap_boing.py` script, which trains Shutter how to play the game with an array of players where each player is assigned an index. After a set amount of epochs to train on, the simulation players are replaced by input from the kinect.  

2. ### Wakeup & Movement

    On wakeup, or start, Shutter moves to an initial position so that it may face players in the frame. This script is called in the `shutter_awaken.py` script. Then, once the game has started, the `shutter_react.py` script will send Shutter to 3 potential locations depending on its predicted move.

3. ### Gesture Recognition

    The gesture recognition node (`interpret_pose.py`) takes in kinect data, and then calculates the angle between one's arms and torso. If the angle in either arm is greater than around 90 degrees, the action is labeled as a _Zip_ or _Zap_. If they are both greater than 90, then the action is a _Boing_.

## Operation

To start a game of Zip Zap Boing, call:

```
cd ~/catkin_ws
source devel/setup.bash
roslaunch pshutter_control play_game.launch
```

If one would like to run each component individually, run the following commands in separate terminals.

Run roscore
```
roscore
```

Run the Shutter Bringup file
```
roslaunch shutter_bringup shutter.launch
```

Enable Motion Commands
```
rosservice call /controller_manager/switch_controller "start_controllers: ['joint_group_controller']
stop_controllers: ['']
strictness: 0
start_asap: false
timeout: 0.0"
```

Start a game of ZZB
```
rosrun pshutter_control zip_zap_boing.py
```

Kinect data pose interpreter
```
rosrun pshutter_control interpret_pose.py
```

Kinect
```
roslaunch azure_kinect_ros_driver driver.launch depth_mode:=WFOV_2X2BINNED body_tracking_enabled:=true
```

Initial Position
```
rosrun pshutter_control shutter_awaken.py
```

Movement Node
```
rosrun pshutter_control shutter_react.py
```

Text to Speech
```
rosrun --prefix "$(rospack find tacotron2_ros)/../.venv/bin/python" tacotron2_ros tacotron2_node.py
```

## Demo
Youtube Video: https://www.youtube.com/watch?v=Gisfcg401Ag&feature=youtu.be
