Dodge the Obstacle - AI Edition 🎮🤖
This repository contains an enhanced Pygame-based dodging game where a player-controlled green square must avoid falling obstacles. The game includes both manual gameplay and an AI-powered reinforcement learning (RL) agent that learns to play using Q-learning.

🚀 Features
Dynamic Obstacles: Random sizes (30-70px), variable speed, acceleration, and horizontal movement.
Smooth Gameplay: Realistic obstacle physics and increasing difficulty over time.
Reinforcement Learning (RL) Agent: A Q-learning agent that learns to dodge obstacles through trial and error.
Customizable Training: Train the AI for as many episodes as desired and watch it improve.
Interactive Modes:
Manual Play: Control the player using arrow keys.
Train AI: Train a Q-learning agent to master the game.
Load AI: Load a pre-trained model and watch the AI play.
🛠 Requirements
pygame
numpy
matplotlib
pickle
Install dependencies with:

bash
Copy
Edit
pip install pygame numpy matplotlib
🎮 How to Play
Run the game in different modes:

bash
Copy
Edit
python main.py
Enter "play" to control the player manually.
Enter "train" to train the RL agent.
Enter "load" to load a pre-trained AI model.
🧠 Training the AI
Train the agent with:

bash
Copy
Edit
python main.py
Then enter "train" and specify the number of episodes.

The agent learns using Q-learning, adjusting its movements to maximize survival time. The trained model is saved as q_table.pkl.

📊 Watch the AI Play
After training, you can let the AI play by entering "load" and watching it in action!
