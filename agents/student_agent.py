# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return random_move(chess_board,player)
  
    def eval(self, board, color, opponent):
        # Simple evaluation function: difference in number of pieces
        player_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        score_diff = player_count - opp_count

        # Add some score penalization for holes in position
        # board[r][c] == 0 indicates empty square

        hole_penalty = 0
        n = board.shape[0] # board.shape[0] indicates board size

        one_tile_dirs = get_directions() # Get all directions (8 directions: up, down, left, right, and diagonals), returns list of tuples
        two_tile_dirs = get_two_tile_directions() # Get all 2-tile directions (16 directions), returns list of tuples
        all_dirs = one_tile_dirs + two_tile_dirs

        # TODO: I am pretty sure these loops can be optimized using memoization 
        for row in range(n):
          for col in range(n):
              
              if board[row][col] != 0:
                  continue # Skip the rest of the inner loop if not an empty tile, DON'T USE BREAK HERE!
                  #Continue goes to next iteration of inner loop, break would exit the entire loop

              near_player = False
              near_opp = False

              for row_distance, col_distance in all_dirs: #For all direction vectors representing move
                  new_row, new_col = row + row_distance, col + col_distance 
                  if 0 <= new_row < n and 0 <= new_col < n: # As long as we are within bounds...
                      if board[new_row][new_col] == color: 
                          near_player = True
                      elif board[new_row][new_col] == opponent:
                          near_opp = True

              # TODO: I am not sure if this considers whose turn it is... might need to adjust
              # I think this is fine, because we will only call this function in mimimax
              if near_opp and not near_player: # If move is reachable  by opponent but not by player
                  hole_penalty += 1

        # TODO: We can add more heuristics here, including a preference for corners, edges, etc.
        # TODO: Modify the weights as needed, this can be done after testing. 
        return score_diff - hole_penalty