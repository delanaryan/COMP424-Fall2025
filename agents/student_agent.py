# Student agent: Add your own agent here
#from matplotlib.pylab import copy
from agents.agent import Agent
import copy
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, get_directions, get_two_tile_directions, MoveCoordinates

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
    # Get all legal moves for the current player
    

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    #return random_move(chess_board,player)
    # Get all legal moves for the current player

    ################################################
    # Alpha-Beta Pruning with Heuristic Evaluation #
    ################################################
    
    legal_moves = get_valid_moves(chess_board, player)

    if not legal_moves:
        return None  # No valid moves available, pass turn

    # Apply heuristic: maximize piece difference, corner control, and minimize opponent mobility
    best_move = None
    best_score = float('-inf')

    for move in legal_moves:
        simulated_board = copy.deepcopy(chess_board)
        #simulated_board = chess_board.copy()

        execute_move(simulated_board, move, player)
        # evaluate by piece difference, corner bonus, and opponent mobility
        move_score = self.eval(simulated_board, player, opponent)

        if move_score > best_score:
            best_score = move_score
            best_move = move

    # Return the best move found (or random fallback)
    return best_move
  
  def isTerminal(self, board : np.ndarray, player : int, opponent : int, depth : int, max_depth : int) -> bool:
    '''
    Check if the game has reached a terminal state or maximum depth.
    '''
    if depth >= max_depth:
        return True
    if check_endgame(board):
        return True
    if not get_valid_moves(board, player) and not get_valid_moves(board, opponent):
        return True
    return False
    
  
  def eval(self, board : np.ndarray, color : int, opponent : int) -> float:
    '''
    Evaluation function to assess board state. Returns a score for the given move.
    '''
    # Simple evaluation function: difference in number of pieces
    # This is taken from greedy_corners_agent.py 
    player_count = np.count_nonzero(board == color)
    opp_count = np.count_nonzero(board == opponent)
    score_diff = player_count - opp_count

    # Add some score penalization for holes in position
    # board[r][c] == 0 indicates empty square

    hole_penalty = self.hole_penalty(board, color, opponent)

    # TODO: We can add more heuristics here, including a preference for corners, edges, etc.
    # We may want to divide our eval function heuristics into separate functions for modularity
    # TODO: Modify the weights as needed, this can be done after testing. 
    return score_diff - 0.5*hole_penalty
  
  def hole_penalty(self, board : np.ndarray, color : int, opponent : int) -> float:
    '''
    Calculate penalty for empty squares (holes) that are adjacent to opponent pieces. Returns score as a float. 
    '''

    penalty = 0
    n = board.shape[0]

    one_tile_dirs = get_directions() # Get all directions (8 directions: up, down, left, right, and diagonals), returns list of tuples
    two_tile_dirs = get_two_tile_directions() # Get all 2-tile directions (16 directions), returns list of tuples
    all_dirs = one_tile_dirs + two_tile_dirs

    for row in range(n):
      for col in range(n):
          
          if board[row][col] != 0:
              continue 

          near_player = False
          near_opp = False

          for row_distance, col_distance in all_dirs: # For all direction vectors representing move
              new_row, new_col = row + row_distance, col + col_distance 
              if 0 <= new_row < n and 0 <= new_col < n: # As long as we are within bounds...
                  if board[new_row][new_col] == color: 
                      near_player = True
                  elif board[new_row][new_col] == opponent:
                      near_opp = True

          # TODO: I am not sure if this considers whose turn it is... might need to adjust
          # I think this is fine, because we will only call this function in mimimax
          if near_opp and not near_player: # If move is reachable by opponent but not by player
              penalty += 1

    return penalty
  