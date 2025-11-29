#!/bin/bash

OUTPUT="tenmoves_output_real.txt"

echo "" > "$OUTPUT" 

echo "Board: Empty"
echo "As player 1:"
python ./simulator2.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/empty_7x7.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"
echo "As player 2:"
python ./simulator2.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/empty_7x7.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"

echo " "
echo " ------ "
echo " "

echo "Board: the circle"
echo "As player 1:"
python ./simulator2.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/the_circle.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"
echo "As player 2:"
python ./simulator2.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/the_circle.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"

echo " "
echo " ------ "
echo " "

echo "Board: big X"
echo "As player 1:"
python ./simulator2.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/big_x.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"
echo "As player 2:"
python ./simulator2.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/big_x.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"

echo " "
echo " ------ "
echo " "

echo "Board: plus 1"
echo "As player 1:"
python ./simulator2.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/plus1.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"
echo "As player 2:"
python ./simulator2.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/plus1.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"

echo " "
echo " ------ "
echo " "

echo "Board: plus 2"
echo "As player 1:"
python ./simulator2.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/plus2.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"
echo "As player 2:"
python ./simulator2.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/plus2.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"

echo " "
echo " ------ "
echo " "

echo "Board: point 4"
echo "As player 1:"
python ./simulator2.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/point4.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"
echo "As player 2:"
python ./simulator2.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/point4.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"

echo " "
echo " ------ "
echo " "

echo "Board: the wall"
echo "As player 1:"
python ./simulator2.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/the_wall.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"
echo "As player 2:"
python ./simulator2.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/the_wall.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"

echo " "
echo " ------ "
echo " "

echo "Board: watch the sides"
echo "As player 1:"
python ./simulator2.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/watch_the_sides.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"
echo "As player 2:"
python ./simulator2.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/watch_the_sides.csv --autoplay --autoplay_runs 10 2>> "$OUTPUT"
