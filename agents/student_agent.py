# Student agent: Add your own agent here
#from matplotlib.pylab import copy
import random
from agents.agent import Agent
import copy
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, get_directions, get_two_tile_directions, MoveCoordinates

POSITION_WEIGHTS = np.array([ # Can change the position weights later
    [ 3, -2,  1,  1,  1, -2,  3], # Weights range from -3 to 3 (from good to bad)
    [-2, -3, -1, -1, -1, -3, -2], # Corners and edges are desireable, middle is neutral
    [ 1, -1,  0,  0,  0, -1,  1], # Diagonally adjacent to corners is risky 
    [ 1, -1,  0,  0,  0, -1,  1],
    [ 1, -1,  0,  0,  0, -1,  1],
    [-2, -3, -1, -1, -1, -3, -2],
    [ 3, -2,  1,  1,  1, -2,  3]
])

TREE = []
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
        simulated_board = deepcopy(chess_board)
        execute_move(simulated_board, move, player)
        minimaxscore = self.minimax(simulated_board, 0, float('-inf'), float('inf'), player, opponent)

        if minimaxscore > best_score:
            best_score = minimaxscore
            best_move = move

    # Return the best move found (or random fallback)
    return best_move or random.choice(legal_moves)

  def minimax(self, chess_board : np.ndarray, depth : int, alpha : float, beta : float, player : int, opponent : int) -> float:
    '''
    Minimax algorithm with alpha-beta pruning.
    Returns the best score for the current player.
    Code repurposed from previous assignment. 
    '''
    legal_moves = get_valid_moves(chess_board, player)

    if not legal_moves:
      return self.eval(chess_board, player, opponent)  # Evaluate current board state, can't return None

    #depth = 0
    #alpha = float('-inf')
    #beta = float('inf')

    node = {
        "depth": depth,
        "player": "MAX" if player == 1 else "MIN",
        "moves": [],
        "alpha": alpha,
        "beta": beta,
        "value": None
    }

    #TREE.append(node)

    if self.isTerminal(chess_board, player, opponent, depth, 10):
        val = self.eval(chess_board, player, opponent)
        node["value"] = val
        TREE.append(node)
        return val
        
    for move in legal_moves:
      #new_board = execute_move(deepcopy(chess_board), move, player)
      new_board = deepcopy(chess_board)
      execute_move(new_board, move, player)

      # changed name of value to child_value
      child_value = self.minimax(new_board, depth + 1, alpha, beta, opponent, player) # swapped player and opponent
      node["moves"].append({
          "move": move,
          "value": child_value,
      })

      #if depth % 2 == 1: # Maximizing player, since depth starts at 0 and max player is at odd depths
      if player == self.my_player_id:    
          alpha = max(alpha, child_value)
      else:
          beta = min(beta, child_value)
      if beta <= alpha:
          break

    node["value"] = alpha if player == self.my_player_id else beta ## Changed from player==1
    TREE.append(node)
    return node["value"]
  
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

    opp_moves = len(get_valid_moves(board, opponent))


    # Add some score penalization for holes in position
    # board[r][c] == 0 indicates empty square

    # Score Difference
    score_diff = player_count - opp_count

    # Hole Penalty
    hole_penalty = self.hole_penalty(board, color, opponent)

    # Mobility Penalty 
    mobility_penalty = -opp_moves

    # Positional Score
    pos_score = self.positional_score(board, color, opponent)

    # Weights? change later? 
    w_score = 1
    w_hole = 0.5
    w_mobility = 0.5
    w_position = 0.5

    # TODO: We can add more heuristics here, including a preference for corners, edges, etc.
    # We may want to divide our eval function heuristics into separate functions for modularity
    # TODO: Modify the weights as needed, this can be done after testing. 
    return (w_score * score_diff) + (w_hole * hole_penalty) + (w_mobility *mobility_penalty) + (w_position * pos_score)
  
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

  def positional_score(self, board, player, opponent):
    player_positions = (board == player) # If the cell it taken by the player
    opponent_positions = (board == opponent) # If the cell is taken by the opp

    player_pos = np.sum(POSITION_WEIGHTS[player_positions]) # Sum of all the positional weights of the player
    opp_pos = np.sum(POSITION_WEIGHTS[opponent_positions]) # Sum of all the pos weights of the opp 

    # Returns the difference: positive if player is in a stronger position and then negative if opponent is
    return player_pos - opp_pos 
  