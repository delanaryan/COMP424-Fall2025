#!/bin/bash

NEWFILE="new_weights.txt"

echo "" > "$NEWFILE" 

echo "Board: big X" >> "$NEWFILE"
echo "------" >> "$NEWFILE"
echo "As player 1:" >> "$NEWFILE"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/big_x.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 mcts_agent --board_path boards/big_x.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "As player 2:"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/big_x.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 mcts_agent --player_2 student_agent --board_path boards/big_x.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "Board: Empty" >> "$NEWFILE"
echo "------" >> "$NEWFILE"
echo "As player 1:" >> "$NEWFILE"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/empty_7x7.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 mcts_agent --board_path boards/empty_7x7.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "As player 2:"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/empty_7x7.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 mcts_agent --player_2 student_agent --board_path boards/empty_7x7.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "Board: plus 1" >> "$NEWFILE"
echo "------" >> "$NEWFILE"
echo "As player 1:" >> "$NEWFILE"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/plus1.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 mcts_agent --board_path boards/plus1.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "As player 2:"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/plus1.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 mcts_agent --player_2 student_agent --board_path boards/plus1.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "Board: plus 2" >> "$NEWFILE"
echo "------" >> "$NEWFILE"
echo "As player 1:" >> "$NEWFILE"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/plus2.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 mcts_agent --board_path boards/plus2.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "As player 2:"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/plus2.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 mcts_agent --player_2 student_agent --board_path boards/plus2.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "Board: point 4" >> "$NEWFILE"
echo "------" >> "$NEWFILE"
echo "As player 1:" >> "$NEWFILE"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/point4.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 mcts_agent --board_path boards/point4.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "As player 2:"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/point4.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 mcts_agent --player_2 student_agent --board_path boards/point4.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "Board: the circle" >> "$NEWFILE"
echo "------" >> "$NEWFILE"
echo "As player 1:" >> "$NEWFILE"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/the_circle.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 mcts_agent --board_path boards/the_circle.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "As player 2:"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/the_circle.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 mcts_agent --player_2 student_agent --board_path boards/the_circle.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "Board: the wall" >> "$NEWFILE"
echo "------" >> "$NEWFILE"
echo "As player 1:" >> "$NEWFILE"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/the_wall.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
     python ./simulator.py --player_1 student_agent --player_2 mcts_agent --board_path boards/the_wall.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "As player 2:"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/the_wall.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 mcts_agent --player_2 student_agent --board_path boards/the_wall.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "Board: watch the sides" >> "$NEWFILE"
echo "------" >> "$NEWFILE"
echo "As player 1:" >> "$NEWFILE"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent --board_path boards/watch_the_sides.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 student_agent --player_2 mcts_agent --board_path boards/watch_the_sides.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done

echo "As player 2:"
for i in {1..5}
do
    echo "Run $i" >> "$NEWFILE"
    echo "Vs Greedy" >> "$NEWFILE"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/watch_the_sides.csv 2>> "$NEWFILE"
    echo "Vs MTCS" >> "$NEWFILE"
    python ./simulator.py --player_1 mcts_agent --player_2 student_agent --board_path boards/watch_the_sides.csv 2>> "$NEWFILE"
    echo "------" >> "$NEWFILE"
    echo " " >> "$NEWFILE"
done