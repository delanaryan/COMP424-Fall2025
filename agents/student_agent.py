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
from agents.greedy_corners_agent import StudentAgent as GreedyAgent

POSITION_WEIGHTS = np.array([ # Can change the position weights later
    [ 3, -2,  1,  1,  1, -2,  3], # Weights range from -3 to 3 (from good to bad)
    [-2, -3, -1, -1, -1, -3, -2], # Corners and edges are desireable, middle is neutral
    [ 1, -1,  0,  0,  0, -1,  1], # Diagonally adjacent to corners is risky 
    [ 1, -1,  0,  0,  0, -1,  1],
    [ 1, -1,  0,  0,  0, -1,  1],
    [-2, -3, -1, -1, -1, -3, -2],
    [ 3, -2,  1,  1,  1, -2,  3]
])

POSITION_WEIGHTS = np.array([ #Position weights normalized on a scale of 0 to 10 for non-negative evaluation
    [10., 1.66666667, 6.66666667, 6.66666667, 6.66666667, 1.66666667, 10.],
    [1.66666667, 0., 3.33333333, 3.33333333, 3.33333333, 0., 1.66666667],
    [6.66666667, 3.33333333, 5., 5., 5., 3.33333333, 6.66666667],
    [6.66666667, 3.33333333, 5., 5., 5., 3.33333333, 6.66666667],
    [6.66666667, 3.33333333, 5., 5., 5., 3.33333333, 6.66666667],
    [1.66666667, 0., 3.33333333, 3.33333333, 3.33333333, 0., 1.66666667],
    [10., 1.66666667, 6.66666667, 6.66666667, 6.66666667, 1.66666667, 10.]
])

MAXDEPTH = 2
#MOVES = {} # Dictionary to hold TERMINAL move scores
TRANSPOSITION_TABLE = {}

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

    best_moves = []

    for move in legal_moves:
      new_board = deepcopy(chess_board)
      execute_move(new_board, move, player)
      #score = self.minimax(new_board, 1, alpha=-float('inf'), beta=float('inf'), player=player, opponent=opponent, maxTurn=True)
      score = self.minimax(new_board, depth=1, alpha=-float('inf'), beta=float('inf'), maximizing_player=player, minimizing_player=opponent, current_mover=opponent)
      
      if score > best_score:
        best_score = score
        best_moves = [move]
      elif score == best_score:
        best_moves.append(move)

    return random.choice(best_moves)

  def minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: int, minimizing_player: int, current_mover: int) -> float:
    """
    Minimax with alpha-beta pruning. `maximizing_player` and `minimizing_player`
    are constants (Option A). `current_mover` is whose turn it is at this node.
    Depth counts plys from the root (root after the initial simulated move had depth=1). 
    """

    # Terminal test: either game ended or depth limit reached. 
    endgame, p1score, p2score = check_endgame(board)

    if depth >= MAXDEPTH or endgame:
      return self.eval_terminal_or_heuristic(board, maximizing_player, minimizing_player, endgame, p1score, p2score)

    # Decide which moves to iterate
    moves = get_valid_moves(board, current_mover)
    if not moves:
      # If no moves available for current mover, the turn would pass to the other player.
      # We'll call minimax with the other player's turn but increment depth by 1 (passing still consumes ply). 
      next_mover = maximizing_player if current_mover == minimizing_player else minimizing_player
      return self.minimax(board, depth+1, alpha, beta, maximizing_player, minimizing_player, next_mover)

    # If it's the maximizing player's turn
    if current_mover == maximizing_player:
      value = -float('inf')
      next_mover = minimizing_player
      for move in moves:
        new_board = deepcopy(board)
        execute_move(new_board, move, current_mover)
        eval_score = self.minimax(new_board, depth+1, alpha, beta, maximizing_player, minimizing_player, next_mover)
        value = max(value, eval_score)
        alpha = max(alpha, eval_score)
        if alpha >= beta:
          break
      return value

    # Else it's the minimizing player's turn
    else:
      value = float('inf')
      next_mover = maximizing_player
      for move in moves:
        new_board = deepcopy(board)
        execute_move(new_board, move, current_mover)
        eval_score = self.minimax(new_board, depth+1, alpha, beta,
                                  maximizing_player, minimizing_player, next_mover)
        value = min(value, eval_score)
        beta = min(beta, eval_score)
        if alpha >= beta:
          break
      return value

  def eval_terminal_or_heuristic(self, board: np.ndarray, maximizing_player: int, minimizing_player: int, endgame: bool, p1score: int, p2score: int) -> float:
    """
    If terminal, give a big positive/negative score relative to maximizing_player.
    Otherwise return the heuristic evaluation. 
    """
    if endgame:
      # Determine player scores
      if p1score > p2score:
        winner = 1
      elif p2score > p1score:
        winner = 2
      else:
        winner = 0  # tie

      if winner == maximizing_player:
        return 10000.0
      elif winner == minimizing_player:
        return -10000.0
      else:
        return 0.0  # tie

    # Not terminal => fallback to heuristic evaluation
    return self.eval(board, maximizing_player, minimizing_player)

  def eval(self, board : np.ndarray, maximizing_player : int, minimizing_player : int) -> float:
    """
    Heuristic evaluation from the perspective of maximizing_player. NEVER overwrites player IDs. 🟢
    Combines piece count difference, mobility (both sides), hole penalty (for squares good for opponent),
    and positional weights.
    """

    max_count = np.count_nonzero(board == maximizing_player)
    min_count = np.count_nonzero(board == minimizing_player)
    score_diff = max_count - min_count  # positive is good for maximizing player

    # Mobility: number of legal moves for each
    max_moves = len(get_valid_moves(board, maximizing_player))
    min_moves = len(get_valid_moves(board, minimizing_player))
    mobility = max_moves - min_moves  # positive is good for maximizing player

    hole_pen = self.hole_penalty(board, maximizing_player, minimizing_player)  # penalty for maximizing_player

    pos_score = self.positional_score(board, maximizing_player, minimizing_player)

    # Game progress to adapt weights (0 = start, 1 = finished)
    empty_count = np.count_nonzero(board == 0)
    total = board.size
    progress = 1.0 - (empty_count / total)

    # Dynamic weights (tuneable)
    w_score     = 5.0 + 10.0 * progress
    w_mobility  = 5.0 - 3.5 * progress
    w_position  = 1.5 - 1.2 * progress
    w_hole      = 0.5 * (1.0 - progress)

    # Combine. Hole_penalty is positive if it's bad for maximizing player so subtract it.
    value = (w_score * score_diff) + (w_mobility * mobility) - (w_hole * hole_pen) + (w_position * pos_score)
    return value
  
  def greedy_eval(self, board : np.ndarray, color: int, opponent: int) -> float:
     return GreedyAgent().evaluate_board(board, color, opponent)
  
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
  