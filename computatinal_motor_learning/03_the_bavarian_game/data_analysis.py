import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

directory = 'participant_data'

# Load board configuration from JSON
board_config = {
    "scoring_left": 85,
    "scoring_top": 176,
    "scoring_width": 1341,
    "scoring_height": 150
}

scoring_left = board_config['scoring_left']
scoring_top = board_config['scoring_top']
scoring_width = board_config['scoring_width']
scoring_height = board_config['scoring_height']




dfs = []

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        dfs.append(pd.read_csv(os.path.join(directory, filename)))


df = dfs[0]
feedback_blocks_series = df['feedback_block']
feedback_trials = {key: [] for key in feedback_blocks_series.keys()}


feedback_blocks = {
    'trajectory': [4, 5, 6],
    'endpos': [7, 8, 9],
    'rl': [10, 11, 12],
    "no feedback": [1, 2, 3]  # Normal feedback type
}


feedback_types = pd.unique(feedback_blocks_series)

feedback_mapping = {}
for feedback_type, block_numbers in feedback_blocks.items():
    for block in block_numbers:
        feedback_mapping[block] = feedback_type

# Map the feedback_blocks_series to the correct feedback type
mapped_feedback_types = feedback_blocks_series.map(feedback_mapping)

# Print the mapped feedback types
print(mapped_feedback_types)


for _, row in df.iterrows():
    x, y, block = row['x'], row['y'], row['feedback_block']
    for fb_type, blocks in feedback_blocks.items():
        if block in blocks:
            current_block =  block % 3
            if fb_type not in feedback_trials:
                feedback_trials[fb_type] = []
            feedback_trials[fb_type].append((x, y, current_block))

print(feedback_trials)
# Define colors for each feedback type
feedback_colors = {
    'trajectory': 'blue',
    'endpos': 'red',
    'rl': 'green',
    "no feedback": 'black'
}

block_mapping = {
    1: "Block 1 (unperturbed)",
    2: "Block 2 (perturbed)",
    3: "Block 3 (unperturbed)"
}

# Define colors (matching the reference image)
dark_green = (0, 100, 0)
light_green = (144, 238, 144)
dark_red = (139, 0, 0)
light_red = (255, 182, 193)


def generate_gradient_image(scoring_rect, start_color, end_color):
    """Generate a gradient effect over a rectangular region."""
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Define gradient for the green to light green
    gradient_green = X  # Left to right
    gradient_red = 1 - X  # Right to left

    # Create an RGBA image with the gradient effect
    gradient_image = np.zeros((100, 100, 3))
    gradient_image[:, :, 0] = gradient_red * (end_color[0] / 255.0)  # Red
    gradient_image[:, :, 1] = gradient_green * (start_color[1] / 255.0)  # Green

    return gradient_image


# Create plot
plt.figure(figsize=(12, 2))

# Generate separate plots for each feedback type
for fb_type in feedback_blocks.keys():
    plt.figure(figsize=(12, 3))

    # Filter trials for the current feedback type
    trials = df[df['feedback_block'].map(feedback_mapping) == fb_type]

    grouped_feedback_block = trials.groupby('feedback_block')[['x', 'y']]

    print(grouped_feedback_block)

    # Plot each trial position
    if not trials.empty:
        plt.scatter(trials['x'], trials['y'], label=fb_type, color=feedback_colors[fb_type], alpha=0.6)

    # Generate gradient background for scoring area
    gradient_image = generate_gradient_image(board_config, dark_green, light_green)
    gradient_image_red = generate_gradient_image(board_config, dark_red, light_red)

    # # Display the green gradient in the upper half
    # plt.imshow(gradient_image, extent=[scoring_left, scoring_left + scoring_width, scoring_top + scoring_height // 2,
    #                                    scoring_top + scoring_height], aspect='auto')
    #
    # # Display the red gradient in the lower half (invert colors)
    # plt.imshow(gradient_image_red,
    #            extent=[scoring_left, scoring_left + scoring_width, scoring_top, scoring_top + scoring_height // 2],
    #            aspect='auto')
# Create plot
plt.figure(figsize=(12, 2))

# Generate separate plots for each feedback type
for fb_type in feedback_blocks.keys():
    plt.figure(figsize=(12, 3))

    # Filter trials for the current feedback type
    trials = df[df['feedback_block'].map(feedback_mapping) == fb_type]

    grouped_feedback_block = trials.groupby('feedback_block')[['x', 'y']]

    print(grouped_feedback_block)

    # Plot each trial position
    if not trials.empty:
        plt.scatter(trials['x'], trials['y'], label=fb_type, color=feedback_colors[fb_type], alpha=0.6)

    # Generate gradient background for scoring area
    gradient_image = generate_gradient_image(board_config, dark_green, light_green)
    gradient_image_red = generate_gradient_image(board_config, dark_red, light_red)

# Draw scoring area boundary
    plt.gca().add_patch(plt.Rectangle(
        (scoring_left, scoring_top),
        scoring_width, scoring_height,
        fill=False, edgecolor='brown', linewidth=3
    ))

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Trial Positions for Feedback Type: {fb_type}")
    plt.legend(title="Feedback Type")
    plt.gca().invert_yaxis()  # Match pygame coordinates
    plt.grid(True)

    # Show the plot for the current feedback type
    plt.show()
    # Draw scoring area boundary
    plt.gca().add_patch(plt.Rectangle(
        (scoring_left, scoring_top),
        scoring_width, scoring_height,
        fill=False, edgecolor='brown', linewidth=3
    ))

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Trial Positions for Feedback Type: {fb_type}")
    plt.legend(title="Feedback Type")
    plt.gca().invert_yaxis()  # Match pygame coordinates
    plt.grid(True)

    # Show the plot for the current feedback type
    plt.show()

