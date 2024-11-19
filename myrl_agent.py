__author__ = "Anthony Deng"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "denan895@student.otago.ac.nz"

import numpy as np
import pickle
import os

agentName = "my sunshine"

# Example of a training specification - in this case it's two sessions,
# one 100 games against two opponents, value_agent and valueplus_agent,
# the other 50 games against random_agent and value_agent. 
training = [ ("value_agent.py", "valueplus_agent.py", 100000),
            ("random_agent.py", "value_agent.py", 100000),
            ("valueplus_agent.py", "random_agent.py", 100000),
            ("my_agent.py", "random_agent.py", 100000)
        ]

# Name of the file to save the agent to.  If you want to retrain your agent
# delete that file.
save_filename="saved_myrl_agent23.pkl"

class RajAgent():
    """
        A class that encapsulates the code dictating the
        behaviour of the agent playing the game of Raj.

        ...

        Attributes
        ----------
        item_values : list of ints
            values of items to bid on
        card_values: list of ints
            cards agent bids with

        Methods
        -------
        AgentFunction(percepts)
            Returns the card value from hand to bid with
        """

    def __init__(self, item_values, card_values):
        """
        :param item_values: list of ints, values of items to bid on
        :card_values: list of ints, cards agent bids with
        """

        self.card_values = card_values
        self.item_values = item_values

        # Initialise a q-table
        self.q_table = {}

        # Initialise learning parameters for q-learning
        self.alpha = 0.3
        self.gamma = 0.99
        self.epsilon = 0.01

        # Initialise variables to store previous state, action, and bank values
        self.prev_state = None
        self.prev_action = None
        self.prev_bank = 0

    """ Load and save function for the agent.
        
        Currently, these load the object properties from a file 
        (all things that are stored in 'self').  You may modify
        them if you need to store more things.

        The load function is called by the engine if the save_file
        is found, the save function is called by the engine after
        the training (which is carried out only if the save file
        hasn't been found)
    """
    def load(self, filename):
        print(f'Loading trained {agentName} agent from {filename}...')
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict) 

    def save(self, filename):
        print(f'Saving trained {agentName} agent to {filename}...')
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()

    def train_start(self):
        """ Invoked once by the engine at the start of the training.

            You may use it to initialise training variables
            You may remove this method if you don't need it.
        """
        self.q_table = {}
        self.alpha = 0.3
        self.gamma = 0.99
        self.epsilon = 0.01

    def train_end(self):
        """ Invoked once by the engine at the start of the training.

            You may use it to finalise training
            You may remove this method if you don't need it.
        """
        print("Training finished!")

    def train_session_start(self):
        """ Invoked by the engine at the start of the training session
                with new opponents (once per tuple in your training variable)

            You may use it to initialise training session against new opponents.
            You may remove this method if you don't need it.
        """
        pass

    def train_session_end(self):
        """ Invoked by the engine at the end of the training session
                with new opponents (once per tuple in your training variable)

            You may use it to finalise training session against 
            You may remove this method if you don't need it.
        """

        pass

    def train_game_start(self):
        """ Invoked by the engine at the start of each game in training

            You may use it to initialise game-specific training variables 
            You may remove this method if you don't need it.
        """
        self.prev_state = None
        self.prev_action = None
        self.prev_bank = 0
        
    def train_game_end(self, banks):
        """ Invoked by the engine at the end of each game training,
            passing in the banks of all players

            Args: banks - a list of integers, the banks of all players at the end of the game
            
            You may remove this method if you don't need it.
        """
        self.prev_state = None
        self.prev_action = None
        self.prev_bank = 0

    def calculate_reward(self, curr_bank, prev_bank, item_val):
        # Calculates the weighted reward
        base_reward = curr_bank - prev_bank
        item_bonus = 0.1 * item_val
        remaining_cards_penalty = -0.05 * len(self.card_values)
        total_reward = base_reward + item_bonus + remaining_cards_penalty

        return total_reward
    
    def is_terminal_state(self, items_left, my_cards):
        # Checks for terminal state
        return len(items_left) == 0 or len(my_cards) == 0

    def handle_terminal_state(self, final_bank):
        if self.prev_state is not None:
            # Handle last q-table update
            terminal_reward = self.calculate_reward(final_bank, self.prev_bank, 0)
            prev_q = self.q_table[self.prev_state][self.prev_action]
            new_q = prev_q + self.alpha * (terminal_reward - prev_q)
            self.q_table[self.prev_state][self.prev_action] = new_q

        self.prev_state = None
        self.prev_action = None
        self.prev_bank = 0

    def AgentFunction(self, percepts):
        """Returns the bid value of the next bid

                :param percepts: a tuple of four items: bidding_on, items_left, my_cards, opponents_cards

                    , where

                    bidding_on - is an integer value of the item to bid on;

                    items_left - the items still to bid on after this bid (the length of the list is the number of
                    bids left in the game)

                    my_cards - the list of cards in the agent's hand

                    bank - total value of items won by this agent in this game
                        
                    opponents_cards - a list of lists of cards in the opponents' hands, so in two player game, this is
                    a list of one list of cards, in three player game, this is a list of two lists, etc.

                :return: value - card value to bid with, must be a number from my_cards
        """

        # Extract different parts of percepts.
        bidding_on = percepts[0]
        items_left = percepts[1]
        my_cards = percepts[2]
        bank = percepts[3]
        opponents_cards = percepts[4:]

        # Checks number of cards left to find terminal state
        if self.is_terminal_state(items_left, my_cards):
            self.handle_terminal_state(bank)
            return my_cards[0]
        else:
            # Create the current state
            state = (tuple(my_cards), bidding_on, bank, opponents_cards)
            # Initialize q-values for unseen states
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(my_cards))

            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(range(len(my_cards)))
            else:
                action = np.argmax(self.q_table[state])
            # Determine the reward for the action
            reward = self.calculate_reward(bank, self.prev_bank, bidding_on)

            # # Update q-values based on the observed state-action-reward
            if self.prev_state is not None:
                if self.prev_state not in self.q_table:
                    self.q_table[self.prev_state] = np.zeros(len(my_cards))
                prev_q = self.q_table[self.prev_state][self.prev_action]
                max_future_q = np.max(self.q_table[state])
                # Update q-value using temporal difference rule
                new_q = prev_q + self.alpha * (reward + self.gamma * max_future_q - prev_q)
                self.q_table[self.prev_state][self.prev_action] = new_q

            # Store the current state and action to be used later 
            self.prev_state = state
            self.prev_action = action
            self.prev_bank = bank

            # Return the bid action
            bid_action = my_cards[action]
            return bid_action
