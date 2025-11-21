#!/bin/bash

OUTPUT="results.txt"

for i in {1..10}
do
    echo "Run $i" >> "$OUTPUT"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent >> "$OUTPUT"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/the_circle.csv >> "$OUTPUT"
python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/big_x.csv >> "$OUTPUT"
python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/plus1.csv >> "$OUTPUT"
python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/plus2.csv >> "$OUTPUT"
python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/plus1.csv >> "$OUTPUT"
python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/the_wall.csv >> "$OUTPUT"
python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent --board_path boards/watch_the_sides.csv >> "$OUTPUT"

echo "Done! Output saved to $OUTPUT"
