__author__ = "Anthony Deng"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<denan895@student.otago.ac.nz>"

import numpy as np

agentName = "minimax"

class RajAgent:
    def __init__(self, item_values, card_values):
        """
        Initialize the agent with the items to bid on and the cards to bid with.

        :param item_values: list of ints, values of items to bid on
        :param card_values: list of ints, cards agent bids with
        """
        self.card_values = card_values
        self.item_values = item_values
        
    def SimulateMove(self, state, card, is_maximising_player):
        new_state = state.copy()
        bidding_on = state['bidding_on']

        if len(new_state['items_left']) > 0:
            new_state['items_left'] = state['items_left'][1:]
            if len(state['items_left']) > 1:
                new_bidding_on = state['items_left'][1]
            else:
                new_bidding_on = None
            new_state['bidding_on'] = new_bidding_on

        if is_maximising_player:
            new_state['my_cards'] = tuple(c for c in state['my_cards'] if c != card)
            new_state['my_bank'].append(bidding_on)
        else:
            new_state['opponent_cards'][0] = tuple(c for c in state['opponent_cards'][0] if c != card)
            new_state['opponent_bank'].append(bidding_on)
        return new_state

    def Terminal(self, state):
        return len(state['items_left']) == 0

    def Evaluate(self, state):
        my_bank_total = sum(state['my_bank'])
        opponent_bank_total = sum(state['opponent_bank'])
        return my_bank_total - opponent_bank_total

    def SimulateAllMoves(self, state, is_maximising_player):
        possible_states = []

        if is_maximising_player:
            for card in state['my_cards']:
                new_state = self.SimulateMove(state, card, is_maximising_player)
                possible_states.append(new_state)
        else:
            for card in state['opponent_cards'][0]:
                new_state = self.SimulateMove(state, card, is_maximising_player)
                possible_states.append(new_state)

        return possible_states
    
    def Minimax(self, state, depth, is_maximising_player):
        if depth == 0 or self.Terminal(state):
            return self.Evaluate(state)

        if is_maximising_player:
            max_eval = float('-inf')  # Initialize the best score as negative infinity
            for new_state in self.SimulateAllMoves(state, True):
                eval = self.Minimax(new_state, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')  # Initialize the best score as positive infinity
            for new_state in self.SimulateAllMoves(state, False):
                eval = self.Minimax(new_state, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    def AgentFunction(self, percepts):
        """
        The main function that returns the agent's bid value.

        :param percepts: a tuple containing the current state of the game
        :return: int, the card value to bid with
        """
        bidding_on = percepts[0]
        items_left = percepts[1]
        my_cards = percepts[2]
        bank = percepts[3]
        opponents_cards = percepts[4:]
    
        # If only one card is left, play it automatically
        if len(percepts[2]) == 1:
            return percepts[2][0]

        state = {
            'bidding_on': bidding_on, 
            'items_left': items_left,
            'my_cards': my_cards,
            'my_bank': [bank],
            'opponent_bank': [],
            'opponent_cards': [opponents_cards]
        }

        best_action = None
        best_value = float('-inf')
        depth = 3

        for card in state['my_cards']:
            new_state = self.SimulateMove(state, card, True)
            move_value = self.Minimax(new_state, depth, is_maximising_player=False)
            if move_value > best_value:
                best_value = move_value
                best_action = card

        return best_action 
