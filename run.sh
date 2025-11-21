#!/bin/bash

OUTPUT="results.txt"

for i in {1..10}
do
    echo "Run $i" >> "$OUTPUT"
    python ./simulator.py --player_1 greedy_corners_agent --player_2 student_agent >> "$OUTPUT"
    python ./simulator.py --player_1 student_agent --player_2 greedy_corners_agent >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

echo "Done! Output saved to $OUTPUT"
