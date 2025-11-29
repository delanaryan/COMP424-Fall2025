from world import World, PLAYER_1_NAME, PLAYER_2_NAME
import argparse
from utils import all_logging_disabled
import logging
import numpy as np
import datetime
import os

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_1", type=str, default="student_agent")
    parser.add_argument("--player_2", type=str, default="random_agent")
    parser.add_argument("--board_path", type=str, default=None,
                        help="Path to the specific board to test")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--display_delay", type=float, default=0.4)
    parser.add_argument("--display_save", action="store_true", default=False)
    parser.add_argument("--display_save_path", type=str, default="plots/")
    parser.add_argument("--autoplay", action="store_true", default=False)
    parser.add_argument("--autoplay_runs", type=int, default=100)
    args = parser.parse_args()
    return args


class Simulator2:
    """
    Simulator2: keeps player order fixed and uses a specific board for autoplay
    """

    def __init__(self, args):
        self.args = args

        # Just a single board for testing
        if self.args.board_path is not None and os.path.isfile(self.args.board_path):
            self.board_options = [self.args.board_path]
        else:
            self.board_options = []

    def reset(self, board_fpath=None):
        """
        Reset the game

        Parameters
        ----------
        board_fpath : str
            if not None, set the board to the layout in the file stored at board_fpath
        """
        if board_fpath is None:
            board_fpath = self.args.board_path

        self.world = World(
            player_1=self.args.player_1,
            player_2=self.args.player_2,
            board_fpath=board_fpath,
            display_ui=self.args.display,
            display_delay=self.args.display_delay,
            display_save=self.args.display_save,
            display_save_path=self.args.display_save_path,
            autoplay=self.args.autoplay,
        )

    def run(self, board_fpath=None):
        self.reset(board_fpath=board_fpath)
        is_end, p0_score, p1_score = self.world.step()
        while not is_end:
            is_end, p0_score, p1_score = self.world.step()
        logger.info(
            f"Run finished. {PLAYER_1_NAME} player, agent {self.args.player_1}: {p0_score}. "
            f"{PLAYER_2_NAME}, agent {self.args.player_2}: {p1_score}"
        )
        return p0_score, p1_score, self.world.p0_time, self.world.p1_time

    def autoplay(self):
        """
        Run multiple simulations on the same board with fixed player order
        """
        p1_win_count = 0
        p2_win_count = 0
        p1_times = []
        p2_times = []

        if self.args.display:
            logger.warning("Since running autoplay mode, display will be disabled")
        self.args.display = False

        if not self.board_options:
            raise ValueError("No valid board found. Please provide --board_path.")

        board_fpath = self.board_options[0]

        with all_logging_disabled():
            for _ in range(self.args.autoplay_runs):
                p0_score, p1_score, p0_time, p1_time = self.run(board_fpath=board_fpath)

                if p0_score > p1_score:
                    p1_win_count += 1
                elif p0_score < p1_score:
                    p2_win_count += 1
                else:  # Tie
                    p1_win_count += 0.5
                    p2_win_count += 0.5

                p1_times.extend(p0_time)
                p2_times.extend(p1_time)

        logger.info(
            f"Player 1, agent {self.args.player_1}, win percentage: {p1_win_count / self.args.autoplay_runs}. "
            f"Maximum turn time was {np.round(np.max(p1_times),5)} seconds."
        )
        logger.info(
            f"Player 2, agent {self.args.player_2}, win percentage: {p2_win_count / self.args.autoplay_runs}. "
            f"Maximum turn time was {np.round(np.max(p2_times),5)} seconds."
        )


if __name__ == "__main__":
    args = get_args()
    simulator = Simulator2(args)
    if args.autoplay:
        simulator.autoplay()
    else:
        simulator.run()
