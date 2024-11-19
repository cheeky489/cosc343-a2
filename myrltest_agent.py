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
            ("valueplus_agent.py", "random_agent.py", 100000) #,
            # ("myrl_agent.py", "random_agent.py", 100000),
            # ("myrl_agent.py", "value_agent.py", 100000),
            # ("myrl_agent.py", "valueplus_agent.py", 100000)
        ]

# Name of the file to save the agent to.  If you want to retrain your agent
# delete that file.
save_filename="saved_myrl_agent3params.pkl"

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
        # self.alpha = 0.5
        # self.gamma = 0.95
        # self.epsilon = 0.2
        self.alpha = 0.3
        self.gamma = 0.99
        self.epsilon = 0.01
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.001
        self.learning_rate_decay = 0.9999

        # Initialise variables to store previous state, action, and bank values
        self.prev_state = None
        self.prev_action = None
        self.prev_bank = 0

        # Performance tracking
        self.games_played = 0
        self.total_reward = 0
        self.win_count = 0
        
        # Advanced features
        self.opponent_models = {}
        self.item_value_estimates = {item: value for item, value in zip(range(len(item_values)), item_values)}
        self.learning_rate_decay = 0.9999

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
        # pass
        # Resets training parameters before training starts
        # self.q_table = {}
        # # self.alpha = 0.5
        # # self.gamma = 0.95
        # # self.epsilon = 0.2
        # self.alpha = 0.3
        # self.gamma = 0.99
        # self.epsilon = 0.01
        self.games_played = 0
        self.total_reward = 0
        self.win_count = 0
        self.opponent_models = {}

    def train_end(self):
        """ Invoked once by the engine at the start of the training.

            You may use it to finalise training
            You may remove this method if you don't need it.
        """
        # pass
        # Lets us know that training has been finished
        # print("Training finished!")
        print(f"Training finished! Games played: {self.games_played}, Win rate: {self.win_count/self.games_played:.2%}")
        print(f"Average reward per game: {self.total_reward/self.games_played:.2f}")
        print("Final item value estimates:", self.item_value_estimates)

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
        # pass
        # Resets game specific variables before each game starts
        self.prev_state = None
        self.prev_action = None
        self.prev_bank = 0
        
    def train_game_end(self, banks):
        """ Invoked by the engine at the end of each game training,
            passing in the banks of all players

            Args: banks - a list of integers, the banks of all players at the end of the game
            
            You may remove this method if you don't need it.
        """
        # pass
        # Resets game specific variables after each game ends
        self.games_played += 1
        if banks[0] == max(banks):
            self.win_count += 1
        self.total_reward += banks[0]
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.alpha *= self.learning_rate_decay

    def get_state(self, bidding_on, items_left, my_cards, bank, opponents_cards):
        return (
            tuple(sorted(my_cards)),
            bidding_on,
            tuple(sorted(items_left)),
            bank,
            tuple(tuple(sorted(hand)) for hand in opponents_cards)
        )

    def update_opponent_model(self, opponent_id, bid, item):
        if opponent_id not in self.opponent_models:
            self.opponent_models[opponent_id] = {'bids': [], 'items': []}
        self.opponent_models[opponent_id]['bids'].append(bid)
        self.opponent_models[opponent_id]['items'].append(item)

    def predict_opponent_bid(self, opponent_id, item):
        if opponent_id not in self.opponent_models or not self.opponent_models[opponent_id]['bids']:
            return np.mean(self.card_values)
        
        opponent_data = self.opponent_models[opponent_id]
        relevant_bids = [bid for bid, bid_item in zip(opponent_data['bids'], opponent_data['items']) if bid_item == item]
        
        if relevant_bids:
            return np.mean(relevant_bids)
        else:
            return np.mean(opponent_data['bids'])

    def update_item_value_estimate(self, item, winning_bid):
        current_estimate = self.item_value_estimates[item]
        self.item_value_estimates[item] = current_estimate * 0.9 + winning_bid * 0.1

    def calculate_bid_utility(self, bid, item, opponents):
        item_value = self.item_value_estimates[item]
        win_probability = 1.0
        for opponent in opponents:
            predicted_opponent_bid = self.predict_opponent_bid(opponent, item)
            if bid > predicted_opponent_bid:
                win_probability *= 0.9
            else:
                win_probability *= 0.1
        return (item_value - bid) * win_probability

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

        # Create the current state as a tuple of relevant percepts
        state = (tuple(my_cards), bidding_on, bank, opponents_cards)

        # Initialize q-values for unseen states
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(my_cards))

        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(range(len(my_cards)))  # Explore: random action
        else:
            action = np.argmax(self.q_table[state])  # Exploit: choose best action

        # Determine the reward based on whether the agent's bank increased or not
        reward = bank - self.prev_bank

        # Update q-values based on the observed state-action-reward
        if self.prev_state is not None: # If previous state exists
            # Ensure prev_state exists in Q-table
            if self.prev_state not in self.q_table:
                # Initialise if missing
                self.q_table[self.prev_state] = np.zeros(len(my_cards))
            # Update q-value using temporal difference rule
            self.q_table[self.prev_state][self.prev_action] += self.alpha * (reward + self.gamma * np.max(self.q_table[state]) - self.q_table[self.prev_state][self.prev_action])
        
        # Store the current state and action to be used later 
        self.prev_state = state
        self.prev_action = action
        self.prev_bank = bank

        # Return the bid action
        return my_cards[action]
