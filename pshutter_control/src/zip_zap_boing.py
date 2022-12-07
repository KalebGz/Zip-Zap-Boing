#!/usr/bin/env python

import random 
import time
import argparse
import numpy as np
from numpy import unravel_index
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
from sklearn.model_selection import train_test_split

# Ros imports
import rospy
from std_msgs.msg import String

#Define a class for each 'move' in the game 
class Action():
    def __init__(self, pointer, action):
        #pointer tells us which player the move is directed at 
        self.pointer = pointer
        #action stores the action that the person takes
        self.action = action
    
    def __repr__(self):
        return "Pointer: " + str(self.pointer) + ", Action: " + str(self.action)

    def __eq__(self, other):
        #we want to define 'equality' (==) for the Move class
        return self.pointer == other.pointer and self.action == other.action

#define a class for a state 'state' in the game
class State():
    def __init__(self, prev_player, curr_player, prev_action):
        self.pp = prev_player
        self.cp = curr_player
        self.pa = prev_action

    def __repr__(self):
        return "Previous Player: " + str(self.pp) + ", Current Player: " + str(self.cp) + ", Action: " + str(self.pa)

    def __eq__(self, other):
        #we want to define 'equality' (==) for the Move class
        return self.pp == other.pp and self.cp == other.cp and self.pa == other.pa

    def __hash__(self):
        return hash((self.pp, self.cp, self.pa))



class ZipZapBoing():
    def __init__(self, N, gameplay="automatic", error=0, errorpass=0):

        self.publisher = rospy.Publisher("/tacotron2/tts", String, queue_size=10)
        self.subscriber = rospy.Subscriber("/zzb_move", String, recieve_player_move ,queue_size=1)
        rospy.init_node("zip_zap_boing", anonymous=False)
        self.rate = rospy.Rate(60)
        #These class variables store the following information:
        #i) The number of players in the game
        #ii) The current player in the game, this is initialised at a random index
        #i) The previous move made
        #iv) The person who made the previous move
        #v) A ledger that holds all of the moves made during the game 
        #vi) arguments as inputed by the user
        #vii) A dicrionary that maps moves to discrete values 
        #iix) This variable dictates the proportion of the time that our players make an incorrect move
        self.N = N
        self.first_player = random.randint(0, N-1)
        self.state = State(self.first_player, self.first_player, 3)
        self.game_ledger = []
        self.gameplay = gameplay
        self.d = {"Zip": 0, "Zap": 1, "Boing": 2}
        self.dd = {0: "Zip", 1: "Zap", 2: "Boing"}
        self.error = error
        self.errorpass = errorpass

        #define the agent that is going to play the game 
        self.agent = BasicRL(self.N, self.game_ledger)

        #define the set of legal moves for all legal contexts that our contextual bandit should receive
        self.contexts = {}
        
        for i in range(N):
            self.contexts[State(i, i, 3)] = None
            self.contexts[State((i+1)%N, i, 0)] = None
            self.contexts[State((i-1)%N, i, 1)] = None
            self.contexts[State((i+1)%N, i, 2)] = None
            self.contexts[State((i-1)%N, i, 2)] = None

        contexts_copy = copy.deepcopy(self.contexts)

        #for each context we want to return all of the legal moves
        for k in contexts_copy:

            pp = k.pp
            cp =k.cp
            pa = k.pa

            #define three legal moves in the game
            Zip = Action( (cp - 1) % N, 0)
            Zap = Action( (cp + 1) % N, 1)
            Boing = Action(pp, 2)

            if pa == 3:
                self.contexts[k] = [Zip, Zap]
            if pa == 0:
                self.contexts[k] = [Zip, Boing]
            if pa == 1:
                self.contexts[k] = [Zap, Boing]
            if pa == 2:
                if (cp - pp) % N == 1:
                    self.contexts[k] = [Zap]
                if (pp - cp) % N == 1:
                    self.contexts[k] = [Zip]

    def print_move(self):
        #the following gives us a visualisation for our game
        arrow_matrix = [f"   {j}   " for j in range(self.N)]

        if self.state.pa == 0:
            arrow_matrix[self.state.pp] = "  <-ZIP-  " 
            print("\n", *arrow_matrix, "\n")
        if self.state.pa == 1:
            arrow_matrix[self.state.pp] = "  -ZAP->  "
            print("\n", *arrow_matrix, "\n")
        if self.state.pa == 2: 
            arrow_matrix[self.state.pp] = "   BOING   "
            print("\n", *arrow_matrix, "\n")
        if self.state.pa == 3: 
            arrow_matrix[self.state.pp] = "   ERROR   "
            print("\n", *arrow_matrix, "\n")

    def request_user_move(self):
        #this function allows a user to manually input moves 
        #user defines the action and the player who the action is directed at
        pointer = int(input("Please write the index of the player you wish to point at:"))
        action = input("Please write the action you would like to take: Zip Zap or Boing")

        #add the users input into an 'Action' variable
        move = Action(pointer, action)

        #check that the user's move is legal, if it is return it, if it isn't request the user move again
        if move in self.contexts[self.state]:
            return move
        else: 
            print("Invalid move, please try again")
            self.request_user_move

    def recieve_player_move(self, msg):
        pass

    def illegal_move(self):
        plausible = True

        if plausible == False:
            #keep choosing random moves until we find one which is illegal
            flag = False

            while flag == False:
                pointer = random.choice(range(self.N))
                action = random.choice([0, 1, 2])
                move = Action(pointer, action)

                if move not in self.contexts[self.state]:
                    flag = True
                    return move

        #alternatively we would like to generate an illegal move from a likely set of illegal moves
        else:
            legal_move = random.choice(self.contexts[self.state])
            all_indexes = [0,1,2]
            plausible_action = all_indexes.pop(legal_move.action)
            move = Action(legal_move.pointer, plausible_action)
            return [legal_move, move]

    def reset_game(self):
        #reset previous sender and previous move to none and attempt to make another move 
        self.state = State(self.state.cp, self.state.cp, 3)

    def make_move(self, speech=False, move=None):

        if self.gameplay == "user":
            next_move = self.request_user_move()


        elif self.gameplay == "automatic":
            #next_move = self.generate_move()

            #we want to make an erroneous move some proportion of time specified by the user 
            error_flag = random.choices([True, False], weights=[self.error, 1- self.error])[0]

            #if the error flag is False then we want to return a correct move, if not we want to return a 
            if error_flag == False:
                next_move= random.choice(self.contexts[self.state])
            else:
                closest_legal_move, next_move = self.illegal_move()


        elif self.gameplay == "participant":
            #make all legal moves unless it is the robots turn, in which case the robot will make a move based on its model
            
            if self.state.cp == 0: #if it is the robot's turn 
                next_move = self.agent.predict_move(self.state)
                print("Model is guessing the next move")

                #for simulation purposes only, we want to check if the move is legal or not 
                if next_move not in self.contexts[self.state]:
                    error_flag = True
                else:
                    error_flag = False

                # TODO output action
                msg = String()

                if not error_flag:
                  msg = "My choice is " + self.dd[next_move.action] + "."
                else:
                  msg = "Error!"

                if speech: self.publisher.publish(msg)

            elif move:

                print(move)
                
                pointer = None
                if move == "Zip":
                    pointer = (self.state.cp - 1) % self.N
                elif move == "Zap":
                    pointer = (self.state.cp + 1) % self.N
                elif move == "Boing":
                    pointer = self.state.pp
                else:
                    return

                next_move = Action(pointer, self.d[move])

                print(next_move)

                if move in self.contexts:
                    error_flag = False
                else:
                    return

                print("Creating Move")

            else: #if it is not the robot's turn make a random legal move choice
                #we want to make an erroneous move some proportion of time specified by the user 
                error_flag = random.choices([True, False], weights=[self.error, 1- self.error])[0]

                #if the error flag is False then we want to return a correct move, if not we want to return a 
                if error_flag == False:
                    next_move= random.choice(self.contexts[self.state])
                else:
                    closest_legal_move, next_move = self.illegal_move()


        if error_flag == False:
            #add the next move to the game ledger
            entry = {"state": self.state, "action": next_move, "reward": 1}
            self.game_ledger.append(entry)
            self.state = State(self.state.cp, next_move.pointer, next_move.action)
        
        else:
            #we want to ignore erroneous move some proportion of time specified by the user 
            errorpass_flag = random.choices([False, True], weights=[self.errorpass, 1- self.errorpass])[0]

            if errorpass_flag == True:
                #if an error is detected we want to restart the game from the person who made the mistake
                entry = {"state": self.state, "action": next_move, "reward": -10} #next entry in the ledger
                self.game_ledger.append(entry)
                self.reset_game()

            else:
                entry = {"state": self.state, "action": next_move, "reward": 1}
                self.game_ledger.append(entry)
                self.state = State(self.state.cp, closest_legal_move.pointer, next_move.action)
        
        #update our model
        self.agent.online_update(entry)

        #print move to the terminal
        self.print_move()

        #optional: wait a second after each move
        self.rate.sleep()
        

class BasicRL():
    def __init__(self, N, ledger):
        self.ledger = ledger
        self.N = N
        self.Reward_matrix = np.zeros((self.N, self.N, 4, self.N, 3))

    def offline_update(self, evaluate = False):
        #constructs the reward matrix, our parameterisation of the reward space, the argument 'evaluate' should be set to true if we want to evaluate the accuracy of our model

        #we need to create a datastructure to parameterise the reward function, for each of the states in the game, we must store an array of rewards over possible acitons
        #we assume no knowledge of legal moves and so the total number of states are N^2 * 3. Because it can be the 'turn' of any of the N players and the player could have
        #passed the turn by and of the N players by either a 'zip', 'zap' or 'boing'.
        #the number of actions is 3 * N -> the number of actions that can be taken (zip, zap, boing) by the number of players the action can be directed at (N).
        #we thus choose to parameterise our reward with an N x N x 3 x N x 3 matrix

        accuracy = []

        for log in self.ledger:
            try:
                #update appropriate index in the reward matrix
                if log['state'].pp == None:
                    pass

                else:
                    #print(log["state"].pp, log["state"].cp, log["state"].pa, log["action"].pointer, log["action"].action)
                    self.Reward_matrix[ log["state"].pp, log["state"].cp, log["state"].pa, log["action"].pointer, log["action"].action ] += log["reward"]

                if evaluate == True:
                    accuracy.append(self.evaluate_reward())

            except: 
                pass

        return self.Reward_matrix, accuracy

    def online_update(self, entry):

        #update the reward matrix with our current entry 
        self.Reward_matrix[ entry["state"].pp, entry["state"].cp, entry["state"].pa, entry["action"].pointer, entry["action"].action ] += entry["reward"]
        

    def evaluate_reward(self):
        #we need a way to randomly sample from all possible states of the game so that we can see how accurate our model is
        game = ZipZapBoing(self.N)
        contexts = game.contexts
        accuracy = 0

        for k in contexts:
            if k.pp == None:
                pass

            predicted_move = self.predict_move(k)
            if predicted_move in contexts[k]:
                accuracy += 1
            
        accuracy = accuracy / len(contexts)

        return accuracy

    def predict_move(self, state):
        #this is the matrix for rewards over the possible actions in our given state 
        actions_matrix = self.Reward_matrix[ state.pp, state.cp, state.pa, :, :]

        #for our policy we want to select the index of the action with the highest reward 
        index = unravel_index(actions_matrix.argmax(), actions_matrix.shape)

        move = Action(index[0], index[1])

        return move

class DeepRL():
    def __init__(self, N, ledger):
        self.N = N
        self.ledger = ledger
        self.input_shape = self.N + self.N + 4
        self.output_shape = (self.N, 4)

    def format_data(self, ledger):

        data = []
        labels = []

        for entry in self.ledger:

            state = entry['state']
            action = entry['action']

            datapoint = np.zeros(self.input_shape)
            label = np.zeros(self.output_shape)

            datapoint[state.pp] = 1
            datapoint[self.N + state.cp] = 1
            datapoint[self.N + self.N + state.pa] = 1

            label[action.pointer, action.action] = 1

            label = [j for sub in label for j in sub]

            data.append(datapoint)
            labels.append(label)
            print(label)

        data = np.array(data)
        labels = np.array(labels)

        data_train, data_test, label_train, label_test = train_test_split(data, labels, train_size=0.8, test_size=0.2)

        return data_train, data_test, label_train, label_test

    def model(self):
        model = tf.keras.Sequential(
            [
                keras.Input(shape = (self.input_shape)),
                layers.Dense(64, activation = 'tanh'),
                layers.Dense(self.N * 4, activation = 'softmax')
            ]
        )
        return model 

    def train(self, model, data_train, labels_train):
        batch_size = 28

        epochs = 100

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(data_train, labels_train, epochs=epochs)

        return None

    def online_update(self, entry):
        return None 

    def predict_move(self, state):
        return None

    def evaluate(self):
        return None

    def main(self):
        return None

g_game = None

#this is the main function in our code 
def play():
    global g_game
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    N = 6
    n = 2000
    gameplay = 'participant'
    error = 0.0
    errorpass = 0.
    learning_regime = 'none'
    evaluate = True

    #plt.title(f"Change in accuracy for different values of error pass, error rate = {error}, number of players = {N}")


    for errorpass in range(1):

        errorpass = errorpass / 10

        # Create new game
        g_game = ZipZapBoing(N, gameplay, error, errorpass)

        # Game loop
        for _ in range(800):

            #make move
            g_game.make_move()

        while not rospy.is_shutdown():
            rospy.spin()

        # Does this need to be reached during gameplay?
        ledger = g_game.game_ledger

        # for item in ledger:
        #   print(item )
        #   print("\n")

        #if the user selects the basic learning regime we want to generate a reward matrix
        if learning_regime == 'basic':
            RLAlg = BasicRL(N, ledger)
            RMatrix, accuracy = RLAlg.offline_update(evaluate=True)

        elif learning_regime == 'deep':
            RLAlg = DeepRL(N, ledger)
            data_train, data_test, labels_train, labels_test = RLAlg.format_data(ledger)

            model = RLAlg.model()

            model.summary()

            print(len(data_train[0]), len(labels_train[0]))

            RLAlg.train(model, data_train, labels_train)

            predictions = model.predict(data_test)

            predictions = [np.reshape(p, (N, 4)) for p in predictions]

            predictions = np.around(predictions, 2)

            for i in range(len(data_test)):
              print(predictions[i])

        #plt.plot(accuracy, label = f"Rate of errorpass = {errorpass}")
    
    #plt.legend()
    #plt.show()

def recieve_player_move(msg):

    if g_game is None: return

    print(f"Move: {msg.data}")

    g_game.make_move(speech=False, move=msg.data)
    if g_game.state.cp == 0:
        g_game.make_move(speech=True)

    time.sleep(3)

if __name__ == "__main__":
    # run the main function
    play()
