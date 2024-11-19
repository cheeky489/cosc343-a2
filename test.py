# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import pickle
from raj import RajGame, Player

__author__ = "Anthony Deng"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "denan895@student.otago.ac.nz"

# Initialize game settings
game_settings = {
   "agentFiles": ("myrl_agent.py", "valueplus_agent.py", "random_agent.py"), # agent files
   "cardValues": (1,2,3,4,5,6),  # value of the cards to bid with
   "itemValues": (-2,-1,1,2,3,4), # values of the items to bid on (must be same length as cardValues)
#    "totalNumberOfGames": 5000,    # total number of games played
   "verbose": True,
   "seed": 0,                     # seed for random choices of bids in the game, None for random seed
}

# Define the test function
def run_tests(num_games=100):
    rnd = np.random.RandomState(game_settings['seed'])

    # Initialize the game
    game = RajGame(
        card_values=game_settings['cardValues'],
        item_values=game_settings['itemValues'],
        num_players=len(game_settings['agentFiles']),
        verbose=game_settings['verbose']
    )

    # Initialize players
    players = []
    player_names = []
    for agentFile in game_settings['agentFiles']:
        try:
            player = Player(game=game, playerFile=agentFile)
            players.append(player)
            player_names.append(player.name)
        except Exception as e:
            print(f"Error initializing player from {agentFile}: {str(e)}")
            return

    # Generate boards
    all_boards = np.zeros((num_games, len(game_settings['itemValues'])))
    for n in range(num_games):
        all_boards[n] = rnd.choice(game_settings['itemValues'], size=len(game_settings['itemValues']), replace=False)

    # Track results
    results = []
    win_counts = np.zeros(len(players))

    # Track score differences
    diff_0_1 = []  # Differences between agent 0 and agent 1
    diff_0_2 = []  # Differences between agent 0 and agent 2
    diff_1_2 = []  # Differences between agent 1 and agent 2

    # Play games and collect results
    for n in range(num_games):
        game_score = game.play(players, items=all_boards[n])
        results.append(game_score)

        # Calculate score differences between agents
        diff_0_1.append(game_score[0] - game_score[1])
        diff_0_2.append(game_score[0] - game_score[2])
        diff_1_2.append(game_score[1] - game_score[2])

        # Determine the winner
        max_score = max(game_score)
        winners = [i for i, score in enumerate(game_score) if score == max_score]
        
        # Increment win count for each winner
        for winner in winners:
            win_counts[winner] += 1

    return results, win_counts, player_names, diff_0_1, diff_0_2, diff_1_2

# Run tests and generate plots
def main():
    num_games = 100000
    results, win_counts, player_names, diff_0_1, diff_0_2, diff_1_2 = run_tests(num_games)

    # Flatten results to make plotting easier
    results = np.array(results)
    mean_scores = np.mean(results, axis=0)

    # Plotting the distribution of score differences between Agent 0 and Agent 1
    plt.figure(figsize=(10, 6))
    plt.hist(diff_0_1, bins=20, alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel(f"Score Difference ({player_names[0]} - {player_names[1]})")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Score Differences ({player_names[0]} vs {player_names[1]})")
    plt.show()

    # Plotting the distribution of score differences between Agent 0 and Agent 2
    plt.figure(figsize=(10, 6))
    plt.hist(diff_0_2, bins=20, alpha=0.75, color='orange', edgecolor='black')
    plt.xlabel(f"Score Difference ({player_names[0]} - {player_names[2]})")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Score Differences ({player_names[0]} vs {player_names[2]})")
    plt.show()

    # Plotting the distribution of score differences between Agent 1 and Agent 2
    plt.figure(figsize=(10, 6))
    plt.hist(diff_1_2, bins=20, alpha=0.75, color='green', edgecolor='black')
    plt.xlabel(f"Score Difference ({player_names[1]} - {player_names[2]})")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Score Differences ({player_names[1]} vs {player_names[2]})")
    plt.show()

    # Plotting win frequencies
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(win_counts)), win_counts, color='green', edgecolor='black')
    plt.xlabel("Player")
    plt.ylabel("Number of Wins")
    plt.title("Win Frequencies of Agents")
    plt.xticks(ticks=range(len(win_counts)), labels=player_names, rotation=45)
    plt.show()

if __name__ == "__main__":
    main()
