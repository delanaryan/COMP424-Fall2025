# Authors: Delana Ryan, Bailey Corriveau, Yuna Ruel
import math
import random
from timeit import default_timer as timer
from typing import Dict, Tuple, List

import numpy as np

from agents.agent import Agent
from store import register_agent
from helpers import (
    get_valid_moves,
    execute_move,
    check_endgame,
    get_directions,
    MoveCoordinates,
    get_two_tile_directions,
)

# Base depth; we adapt this depending on how many empty squares remain.
BASE_MAX_DEPTH = 3
TERMINAL_SCORE_MULTIPLIER = 1000

# Positional weights matrix (borrowed / adapted from original student_agent)
POSITION_WEIGHTS = np.array([
    [ 3, -2,  1,  1,  1, -2,  3],
    [-2, -3, -1, -1, -1, -3, -2],
    [ 1, -1,  1,  0,  1, -1,  1],
    [ 1, -1,  0,  3,  0, -1,  1],
    [ 1, -1,  1,  0,  1, -1,  1],
    [-2, -3, -1, -1, -1, -3, -2],
    [ 3, -2,  1,  1,  1, -2,  3],
], dtype=float)


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Strong search-based Ataxx agent.

    This version combines:
    - minimax + alpha–beta
    - a transposition table for caching
    - move ordering
    - a rich heuristic (piece count, mobility, hole penalty, positional weights)
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "StudentAgent"

        # transposition table: state -> value
        self.transposition_table: Dict[Tuple[bytes, int, int, int], float] = {}

        # Precompute neighbour directions (8 surrounding squares)
        self._neighbour_dirs: List[Tuple[int, int]] = get_directions()

    # ------------------------------------------------------------------
    # Entry point called by the framework
    # ------------------------------------------------------------------
    def step(self, chess_board: np.ndarray, player: int, opponent: int):
        # Clear cache for this move.
        #self.transposition_table.clear()

        start = timer()

        legal_moves = get_valid_moves(chess_board, player)
        if not legal_moves: 
            raise RuntimeError("StudentAgent.step called with no legal moves.")

        max_depth = self.choose_search_depth(chess_board)

        # Order moves at the root using a one-ply heuristic.
        ordered_moves = self.order_moves(chess_board, legal_moves, player, opponent)

        best_score = -math.inf
        best_moves: List[MoveCoordinates] = []

        time = 0 
 
        for move in ordered_moves:
            if time <= 1.4:
                new_board = chess_board.copy()
                execute_move(new_board, move, player)

                score = self.minimax(
                    board=new_board,
                    depth=1,
                    max_depth=max_depth,
                    current_player=opponent,
                    root_player=player,
                    other_player=opponent,
                    alpha=-math.inf,
                    beta=math.inf,
                )

                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

                end = timer()
                time = end - start

        return random.choice(best_moves)

    # ------------------------------------------------------------------
    # Minimax with alpha–beta
    # ------------------------------------------------------------------
    def minimax(
        self,
        board: np.ndarray,
        depth: int,
        max_depth: int,
        current_player: int,
        root_player: int,
        other_player: int,
        alpha: float,
        beta: float,
    ) -> float:
        time = timer()
        # Terminal state?
        endgame, p1_score, p2_score = check_endgame(board)
        if endgame:
            if root_player == 1:
                diff = p1_score - p2_score
            else:
                diff = p2_score - p1_score
            return diff * TERMINAL_SCORE_MULTIPLIER

        # Depth limit reached -> heuristic evaluation.
        if depth >= max_depth:
            return self.evaluate(board, root_player, other_player, depth, max_depth)
        
        # Transposition table lookup.
        depth_remaining = max_depth - depth
        key = (board.tobytes(), current_player, depth_remaining, root_player)
        cached = self.transposition_table.get(key)
        if cached is not None:
            return cached

        moves = get_valid_moves(board, current_player)

        # Handle "pass" when current_player has no moves.
        if not moves:
            next_player = 1 if current_player == 2 else 2
            other_moves = get_valid_moves(board, next_player)
            if not other_moves:
                # No-one can move: static evaluation.
                return self.evaluate(board, root_player, other_player, depth, max_depth)
            return self.minimax(
                board=board,
                depth=depth + 1,
                max_depth=max_depth,
                current_player=next_player,
                root_player=root_player,
                other_player=other_player,
                alpha=alpha,
                beta=beta,
            )

        maximizing = (current_player == root_player)

        if maximizing:
            value = -math.inf
            ordered = self._order_child_moves(board, moves, current_player, other_player)
            for move in ordered:
                child_board = board.copy()
                execute_move(child_board, move, current_player)

                value = max(
                    value,
                    self.minimax(
                        board=child_board,
                        depth=depth + 1,
                        max_depth=max_depth,
                        current_player=1 if current_player == 2 else 2,
                        root_player=root_player,
                        other_player=other_player,
                        alpha=alpha,
                        beta=beta,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            ordered = self._order_child_moves(board, moves, current_player, other_player, reverse=True)
            for move in ordered:
                child_board = board.copy()
                execute_move(child_board, move, current_player)

                value = min(
                    value,
                    self.minimax(
                        board=child_board,
                        depth=depth + 1,
                        max_depth=max_depth,
                        current_player=1 if current_player == 2 else 2,
                        root_player=root_player,
                        other_player=other_player,
                        alpha=alpha,
                        beta=beta,
                    ),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break

        # Store in cache and return.
        self.transposition_table[key] = value
        return value

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------
    def order_moves(
        self,
        board: np.ndarray,
        moves: List[MoveCoordinates],
        player: int,
        opponent: int,
    ) -> List[MoveCoordinates]:
        """
        Order moves at the root using a one-ply heuristic.
        """
        scored: List[Tuple[float, MoveCoordinates]] = []
        for move in moves:
            temp = board.copy()
            execute_move(temp, move, player)
            score = self.evaluate(temp, player, opponent, depth=0, max_depth=1)
            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for (_, m) in scored]

    def _order_child_moves(
        self,
        board: np.ndarray,
        moves: List[MoveCoordinates],
        player: int,
        opponent: int,
        reverse: bool = False,
    ) -> List[MoveCoordinates]:
        """
        Cheaper move ordering for internal nodes: prefer moves that
        gain discs and move to corners / edges.
        """
        scored: List[Tuple[float, MoveCoordinates]] = []
        n = board.shape[0]

        for move in moves:
            src_r, src_c = move.get_src()
            dst_r, dst_c = move.get_dest()

            # Positional bonus.
            is_corner = (dst_r in (0, n - 1)) and (dst_c in (0, n - 1))
            is_edge = (
                (dst_r == 0 or dst_r == n - 1)
                or (dst_c == 0 or dst_c == n - 1)
            )
            pos_bonus = 0.0
            if is_corner:
                pos_bonus = 3.0
            elif is_edge:
                pos_bonus = 1.0

            # Local disc gain (approximate number of opponent discs flipped).
            gain = 0
            for dr, dc in self._neighbour_dirs:
                rr, cc = dst_r + dr, dst_c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    if board[rr, cc] == opponent:
                        gain += 1

            # Jump vs duplication: jumps lose the source disc.
            dist = max(abs(dst_r - src_r), abs(dst_c - src_c))
            if dist == 2:
                gain -= 1

            score = gain + pos_bonus
            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=not reverse)
        return [m for (_, m) in scored]

    # ------------------------------------------------------------------
    # Evaluation function and helpers
    # ------------------------------------------------------------------
    def evaluate(self, board, me, opp, depth, max_depth):
        """
        Fast + strong evaluation:
        - disc difference
        - light mobility (only for us, only shallow depth)
        - light hole penalty (only shallow depth)
        - positional weights
        """

        # Disc difference
        my_count = np.count_nonzero(board == me)
        opp_count = np.count_nonzero(board == opp)
        score_diff = my_count - opp_count

        # Positional score (cheap)
        pos_score = self.positional_score(board, me, opp)

        # Mobility (expensive) → only compute near root
        if depth <= 1:
            my_moves = len(get_valid_moves(board, me))
            mobility = my_moves
        else:
            mobility = 0

        # Hole penalty (expensive) → only compute near root
        if depth <= 1:
            hole_pen = self.hole_penalty(board, me, opp)
        else:
            hole_pen = 0

        # Game progress
        empty_count = np.count_nonzero(board == 0)
        progress = 1.0 - (empty_count / board.size) # 0 = start, 1 = end game

        # Dynamic weights
        w_score    = 5.0 + 10.0 * progress
        w_mobility = 4.0 - 3.0 * progress
        w_position = 2.0 - 1.5 * progress
        w_hole     = 0.5 * (1.0 - progress)

        value = (
            w_score * score_diff +
            w_mobility * mobility +
            w_position * pos_score -
            w_hole * hole_pen
        )

        # Slight horizon factor
        return value * (1.0 + 0.03 * (depth / max_depth))

    def hole_penalty(self, board: np.ndarray, color: int, opponent: int) -> float:
        """
        Penalty for empty squares that are good for opponent:
        they are near opponent discs but not near our discs.
        """
        penalty = 0.0
        n = board.shape[0]

        one_tile_dirs = get_directions()
        two_tile_dirs = get_two_tile_directions()
        all_dirs = one_tile_dirs + two_tile_dirs

        for row in range(n):
            for col in range(n):
                if board[row, col] != 0:
                    continue  # only consider empty squares

                near_player = False
                near_opp = False

                for dr, dc in all_dirs:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        if board[nr, nc] == color:
                            near_player = True
                        elif board[nr, nc] == opponent:
                            near_opp = True

                if near_opp and not near_player:
                    penalty += 1.0

        return penalty

    def positional_score(self, board: np.ndarray, player: int, opponent: int) -> float:
        """
        Positional score based on POSITION_WEIGHTS.
        Positive if `player` is better placed than `opponent`.
        """
        weights = POSITION_WEIGHTS

        player_positions = (board == player)
        opponent_positions = (board == opponent)

        player_pos = float(np.sum(weights[player_positions]))
        opp_pos = float(np.sum(weights[opponent_positions]))

        return player_pos - opp_pos

    def choose_search_depth(self, board: np.ndarray) -> int:
        """
        Pick a search depth based on how many empty squares remain.
        We keep this slightly shallower than a super-deep search because the
        heuristic is more expensive than a simple one.
        """
        empty_count = np.count_nonzero(board == 0)

        if empty_count > 20:
            # Very early game: branching factor is huge.
            return BASE_MAX_DEPTH        # 3
        elif empty_count > 8:
            # Mid game.
            return BASE_MAX_DEPTH - 1    # 4
        else:
            # Late game: fewer moves, can afford deeper search.
            return BASE_MAX_DEPTH + 1    # 5