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

POSITION_WEIGHTS = np.array([ #Position weights normalized on a scale of 0 to 10 for non-negative evaluation
    [10., 1.66666667, 6.66666667, 6.66666667, 6.66666667, 1.66666667, 10.],
    [1.66666667, 0., 3.33333333, 3.33333333, 3.33333333, 0., 1.66666667],
    [6.66666667, 3.33333333, 5., 5., 5., 3.33333333, 6.66666667],
    [6.66666667, 3.33333333, 5., 5., 5., 3.33333333, 6.66666667],
    [6.66666667, 3.33333333, 5., 5., 5., 3.33333333, 6.66666667],
    [1.66666667, 0., 3.33333333, 3.33333333, 3.33333333, 0., 1.66666667],
    [10., 1.66666667, 6.66666667, 6.66666667, 6.66666667, 1.66666667, 10.]
])

MAXDEPTH = 6
#MOVES = {} # Dictionary to hold TERMINAL move scores
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
        return None 

    best_score = -float('inf')
    best_move = None

    for move in legal_moves:
      new_board = np.copy(chess_board) # np.copy instead of deep copy 
      execute_move(new_board, move, player)
      score = self.minimax(new_board, depth=1, alpha=-float('inf'), beta=float('inf'), player=player, opponent=opponent, maxTurn=False) # its mins turn

      if score > best_score:
          best_score = score
          best_move = move

    return best_move

  def minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, player: int, opponent: int, maxTurn: bool) -> float:
    """
    Minimax with alpha-beta pruning.
    Code based off algorithm provided by Geeks for Geeks: https://www.geeksforgeeks.org/dsa/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/
    """

    if self.isTerminal(board, player, opponent, depth): # removed depth + 1
        return self.eval(board, player, opponent)

        # Determine whose turn it is
        cur_player = player if maxTurn else opponent
        valid_moves = get_valid_moves(board, cur_player)

        # Move ordering
        valid_moves = self.order_moves(board, valid_moves, cur_player, opponent if maxTurn else player)

    if maxTurn: 
        max_eval = -float('inf')
        for move in valid_moves:
            new_board = np.copy(board)
            execute_move(new_board, move, cur_player)
            eval_score = self.minimax(new_board, depth+1, alpha, beta, player, opponent, False)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break 
        return max_eval
    
    else: 
        min_eval = float('inf')
        for move in valid_moves:
            new_board = np.copy(board)
            execute_move(new_board, move, cur_player)
            eval_score = self.minimax(new_board, depth + 1, alpha, beta, player, opponent, True)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break 
        return min_eval
  
  def isTerminal(self, board : np.ndarray, player : int, opponent : int, depth : int) -> bool:
    '''
    Check if the game has reached a terminal state or maximum depth.
    '''
    if depth >= MAXDEPTH:
        return True
    if check_endgame(board):
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
    mobility_penalty = len(get_valid_moves(board, color)) - len(get_valid_moves(board, opponent))

    # Positional Score
    pos_score = self.positional_score(board, color, opponent)

    # Perimeter Penalty
    perim_penalty = -(self.perimeter_penalty(board, color, opponent))

    # For the progress tracking
    empty_count = np.count_nonzero(board == 0)
    total = board.size
    progress = 1 - (empty_count/total)  # 0 = start, 1 = end game
    
    # Dynamic weights across game phases
    w_score     = 5.0 + 10.0*progress
    w_mobility  = 5.0 - 3.5*progress
    w_position  = 1.5 - 1.2*progress
    w_hole      = 0.5 * (1 - progress)
    w_perimeter = 1.0 * (1 + progress)

    evaluation = ((w_score * score_diff) + (w_hole * hole_penalty) + (w_mobility * mobility_penalty)
                + (w_position * pos_score) + (w_perimeter * perim_penalty))

    # TODO: We can add more heuristics here, including a preference for corners, edges, etc.
    # We may want to divide our eval function heuristics into separate functions for modularity
    # TODO: Modify the weights as needed, this can be done after testing. 
    return evaluation
  
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

  def positional_score(self, board : np.ndarray, player : int, opponent : int) -> float:
    """
    Scoring based off of what placements are more desireable
    """
    player_positions = (board == player) # If the cell it taken by the player
    opponent_positions = (board == opponent) # If the cell is taken by the opp

    player_pos = np.sum(POSITION_WEIGHTS[player_positions]) # Sum of all the positional weights of the player
    opp_pos = np.sum(POSITION_WEIGHTS[opponent_positions]) # Sum of all the pos weights of the opp 

    # Returns the difference: positive if player is in a stronger position and then negative if opponent is
    return player_pos - opp_pos 

  def perimeter_penalty(self, board : np.ndarray, player : int, opponent : int) -> float:
    """
    player's pieces adjacent to empty squares
    """
    n = board.shape[0]
    dirs = get_directions()
    count = 0

    for row in range(n):
        for col in range(n):
            if board[row][col] != player:
                continue
            for dr, dc in dirs:
                nr, nc = row + dr, col + dc
                if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == 0:
                    count += 1
                    break
    return count

  def order_moves(self, board : np.ndarray, valid_moves : list, cur_player : int, other_player : int) -> list:
    """
    Move ordering = prefer moves that capture lots of opponent pieces and that increase mobility
    """
    ordered = []

    for move in moves:
        # Simulate
        new_board = np.copy(board)
        execute_move(new_board, move, cur_player)

        # Capture estimate, difference in opponent piece count (so positive if we gained)
        capture_gain = np.count_nonzero(board == other_player) - np.count_nonzero(new_board == other_player)
        # Mobility estimate, how many moves we have after making the move
        mobility = len(get_valid_moves(new_board, cur_player))

        score = (capture_gain*2.0) + (mobility*0.5) # Combine the estimates
        scores.append((score, move))

    # Best moves first
    scores.sort(key=lambda x: x[0], reverse=True)
    ordered_moves = [m for (_, m) in scores]
    return ordered_moves