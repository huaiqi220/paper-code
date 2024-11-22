import os
import random



# Path to the original label file and the output file
label_file_path = "/home/hi/zhuzi/data/mpii_final/Label/labels.txt"
output_file_path = "/home/hi/zhuzi/data/mpii_final/Label/processed_labels.txt"


# Read all lines from the original label file
with open(label_file_path, 'r') as file:
    lines = file.readlines()

# Process each line by appending a randomly selected other line
processed_lines = []
for line in lines:
    # Remove newline characters from the current line
    line = line.strip()
    
    # Randomly choose another line that is different from the current line
    random_line = line
    while random_line == line:
        random_line = random.choice(lines).strip()
    
    # Concatenate the current line with the randomly selected line
    processed_line = line + " " + random_line
    processed_lines.append(processed_line)

# Write the processed lines to the new output file
with open(output_file_path, 'w') as file:
    for processed_line in processed_lines:
        file.write(processed_line + '\n')

print("Processing completed and saved to:", output_file_path)
