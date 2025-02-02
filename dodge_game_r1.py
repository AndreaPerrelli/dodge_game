import pygame
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ---------------------------
# Global Constants and Config
# ---------------------------
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
PLAYER_SIZE = 50
FPS = 30

# ---------------------------
# Game Class (Human Play)
# ---------------------------
class Game:
    """
    A Pygame-based dodging game where the player-controlled green square must avoid falling obstacles.
    Enhancements include:
      - Obstacles with variable sizes (30-70 pixels)
      - Randomized vertical speeds and acceleration (speed increases as they fall)
      - Horizontal movement (bouncing off screen edges)
      - Multiple obstacles (each new obstacle is spawned when one goes off-screen)
    """
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Enhanced Dodge the Obstacle!")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.reset_game()

    def reset_game(self):
        self.player_x = WINDOW_WIDTH // 2 - PLAYER_SIZE // 2
        self.player_y = WINDOW_HEIGHT - PLAYER_SIZE - 10
        self.obstacles = []
        # Start with one obstacle; more will be added as the game goes on.
        self.spawn_obstacle()
        self.score = 0
        self.game_over = False

    def spawn_obstacle(self):
        # Generate obstacle parameters.
        size = random.randint(30, 70)
        x = random.randint(0, WINDOW_WIDTH - size)
        y = -size  # start just above the screen
        speed = random.uniform(3, 7)
        acceleration = random.uniform(0.1, 0.3)
        move_x = random.choice([-2, -1, 0, 1, 2])  # horizontal movement speed
        # Each obstacle is represented as a dictionary.
        obstacle = {
            'x': x,
            'y': y,
            'size': size,
            'speed': speed,
            'acceleration': acceleration,
            'move_x': move_x
        }
        self.obstacles.append(obstacle)

    def handle_input(self):
        # Process events so the window remains responsive.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # Use arrow keys to move left/right.
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player_x -= 10
        if keys[pygame.K_RIGHT]:
            self.player_x += 10
        self.player_x = max(0, min(self.player_x, WINDOW_WIDTH - PLAYER_SIZE))

    def update_obstacles(self):
        # Update each obstacle.
        for obs in self.obstacles:
            # Update vertical position and speed (simulate acceleration).
            obs['y'] += obs['speed']
            obs['speed'] += obs['acceleration']
            # Update horizontal position.
            obs['x'] += obs['move_x']
            # Bounce horizontally if hitting the edges.
            if obs['x'] < 0:
                obs['x'] = 0
                obs['move_x'] = -obs['move_x']
            if obs['x'] + obs['size'] > WINDOW_WIDTH:
                obs['x'] = WINDOW_WIDTH - obs['size']
                obs['move_x'] = -obs['move_x']
        
        # Remove obstacles that have passed the bottom and spawn a new one.
        for obs in self.obstacles.copy():
            if obs['y'] > WINDOW_HEIGHT:
                self.obstacles.remove(obs)
                self.score += 1
                self.spawn_obstacle()

    def check_collision(self):
        # Create a rectangle for the player.
        player_rect = pygame.Rect(self.player_x, self.player_y, PLAYER_SIZE, PLAYER_SIZE)
        # Check collision with each obstacle.
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['x'], obs['y'], obs['size'], obs['size'])
            if player_rect.colliderect(obs_rect):
                self.game_over = True

    def draw(self):
        self.screen.fill((0, 0, 0))
        # Draw the player.
        pygame.draw.rect(self.screen, (0, 255, 0), (self.player_x, self.player_y, PLAYER_SIZE, PLAYER_SIZE))
        # Draw obstacles.
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (255, 0, 0), (obs['x'], obs['y'], obs['size'], obs['size']))
        score_text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()

    def run(self):
        # Main game loop.
        while not self.game_over:
            self.clock.tick(FPS)
            self.handle_input()
            self.update_obstacles()
            self.check_collision()
            self.draw()
        # Display Game Over screen.
        self.screen.fill((0, 0, 0))
        game_over_text = self.font.render("Game Over! Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(game_over_text, (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2))
        pygame.display.flip()
        pygame.time.wait(2000)
        pygame.quit()

# ---------------------------
# Game Environment for RL
# ---------------------------
class GameEnv:
    """
    A simplified gym-style environment for RL training.
    The environment now uses enhanced difficulty mechanics:
      - Variable obstacle size, randomized initial speed and acceleration.
      - Horizontal movement (with reflection at boundaries).
    The state is a 5-tuple:
      (player_bin, obs_x_bin, obs_y_bin, size_bin, move_x_bin)
      
    Actions:
        0: move left (by 20 pixels)
        1: stay
        2: move right (by 20 pixels)
        
    Rewards:
        +1 for surviving each step, -100 on collision.
    """
    def __init__(self, render=False):
        self.render = render
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("RL Agent Playing!")
            self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        # Reset player position.
        self.player_x = WINDOW_WIDTH // 2 - PLAYER_SIZE // 2
        self.player_y = WINDOW_HEIGHT - PLAYER_SIZE - 10
        
        # Create a new obstacle with enhanced parameters.
        self.obs_size = random.randint(30, 70)
        self.obs_x = random.randint(0, WINDOW_WIDTH - self.obs_size)
        self.obs_y = -self.obs_size
        self.obstacle_speed = random.uniform(3, 7)
        self.obstacle_acceleration = random.uniform(0.1, 0.3)
        self.obs_move_x = random.choice([-2, -1, 0, 1, 2])
        
        self.score = 0
        self.steps = 0
        self.done = False
        return self.get_state()
    
    def step(self, action):
        # Action: 0 = left, 1 = stay, 2 = right.
        if action == 0:
            self.player_x -= 20
        elif action == 2:
            self.player_x += 20
        self.player_x = max(0, min(self.player_x, WINDOW_WIDTH - PLAYER_SIZE))
        
        # Update the obstacle.
        self.obs_y += self.obstacle_speed
        self.obstacle_speed += self.obstacle_acceleration
        self.obs_x += self.obs_move_x
        # Bounce horizontally if hitting boundaries.
        if self.obs_x < 0:
            self.obs_x = 0
            self.obs_move_x = -self.obs_move_x
        if self.obs_x + self.obs_size > WINDOW_WIDTH:
            self.obs_x = WINDOW_WIDTH - self.obs_size
            self.obs_move_x = -self.obs_move_x
        
        self.steps += 1
        reward = 1  # small reward for surviving a step
        
        # Check collision.
        player_rect = pygame.Rect(self.player_x, self.player_y, PLAYER_SIZE, PLAYER_SIZE)
        obs_rect = pygame.Rect(self.obs_x, self.obs_y, self.obs_size, self.obs_size)
        if player_rect.colliderect(obs_rect):
            reward = -100
            self.done = True
        
        # If the obstacle goes off the bottom, respawn it and increase score.
        if self.obs_y > WINDOW_HEIGHT:
            self.score += 1
            # Respawn obstacle with new random parameters.
            self.obs_size = random.randint(30, 70)
            self.obs_x = random.randint(0, WINDOW_WIDTH - self.obs_size)
            self.obs_y = -self.obs_size
            self.obstacle_speed = random.uniform(3, 7)
            self.obstacle_acceleration = random.uniform(0.1, 0.3)
            self.obs_move_x = random.choice([-2, -1, 0, 1, 2])
        
        if self.render:
            self.render_env()
        return self.get_state(), reward, self.done, {}
    
    def get_state(self):
        # Discretize state into bins.
        cell_size = PLAYER_SIZE  # 50 pixels
        player_bin = self.player_x // cell_size
        obs_x_bin = self.obs_x // cell_size
        obs_y_bin = max(0, min(int(self.obs_y) // cell_size, WINDOW_HEIGHT // cell_size))
        
        # Bin obstacle size: 0 (small), 1 (medium), 2 (large)
        if self.obs_size < 40:
            size_bin = 0
        elif self.obs_size < 60:
            size_bin = 1
        else:
            size_bin = 2
        
        # Bin horizontal movement: 0 (moving left), 1 (stationary), 2 (moving right)
        if self.obs_move_x < 0:
            move_x_bin = 0
        elif self.obs_move_x == 0:
            move_x_bin = 1
        else:
            move_x_bin = 2
        
        return (player_bin, obs_x_bin, obs_y_bin, size_bin, move_x_bin)
    
    def render_env(self):
        # Process events so the window remains responsive.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                pygame.quit()
                sys.exit()
        self.screen.fill((0, 0, 0))
        # Draw the player.
        pygame.draw.rect(self.screen, (0, 255, 0), (self.player_x, self.player_y, PLAYER_SIZE, PLAYER_SIZE))
        # Draw the obstacle.
        pygame.draw.rect(self.screen, (255, 0, 0), (self.obs_x, self.obs_y, self.obs_size, self.obs_size))
        font = pygame.font.SysFont("Arial", 24)
        score_text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def close(self):
        if self.render:
            pygame.quit()

# ---------------------------
# Q-Learning Agent
# ---------------------------
class QLearningAgent:
    """
    A simple Q-learning agent that uses a dictionary-based Q-table.
    The agent uses an epsilon-greedy policy for exploration.
    """
    def __init__(self, actions, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.q_table = {}  # Maps state tuples to arrays of Q-values (one per action)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.actions = actions  # e.g., [0, 1, 2]
    
    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]
    
    def choose_action(self, state):
        qs = self.get_qs(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return int(np.argmax(qs))
    
    def update(self, state, action, reward, next_state):
        qs = self.get_qs(state)
        next_qs = self.get_qs(next_state)
        best_next_q = np.max(next_qs)
        qs[action] = qs[action] + self.alpha * (reward + self.gamma * best_next_q - qs[action])
        self.q_table[state] = qs
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename="q_table.pkl"):
        """
        Save the Q-table to a file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename="q_table.pkl"):
        """
        Load the Q-table from a file.
        """
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from {filename}")

# ---------------------------
# Training Loop for the RL Agent
# ---------------------------
def train_agent(episodes=500):
    """
    Train the Q-learning agent for a specified number of episodes.
    After training, the Q-table is saved to disk.
    """
    env = GameEnv(render=False)
    agent = QLearningAgent(actions=[0, 1, 2])
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            step_count += 1
            if done:
                break
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward} | Steps: {step_count} | Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward over Episodes")
    plt.show()
    
    # Save the trained model
    agent.save_model("q_table.pkl")
    
    return agent
# ---------------------------
# Run Trained Agent (with Rendering)
# ---------------------------
def run_agent(agent, episodes=5):
    """
    Let the trained agent play the game with rendering so you can watch its performance.
    """
    env = GameEnv(render=True)
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")
                pygame.time.wait(1000)
                break
    env.close()

# ---------------------------
# Main Function: Choose Mode
# ---------------------------
def main():
    """
    Choose between manual play ("play"), training the RL agent ("train"), or
    loading an existing trained model ("load").
    """
    mode = input("Enter 'play' to play the game manually, 'train' to train the AI, or 'load' to load a trained model: ").strip().lower()
    
    if mode == 'play':
        game = Game()
        game.run()
    elif mode == 'train':
        try:
            episodes = int(input("Enter number of training episodes (e.g., 500): "))
        except ValueError:
            episodes = 500
        agent = train_agent(episodes)
        test_mode = input("Do you want to watch the trained agent play? (y/n): ").strip().lower()
        if test_mode == 'y':
            run_agent(agent)
    elif mode == 'load':
        # Create a new agent and load the saved Q-table
        agent = QLearningAgent(actions=[0, 1, 2])
        agent.load_model("q_table.pkl")
        test_mode = input("Do you want to watch the loaded agent play? (y/n): ").strip().lower()
        if test_mode == 'y':
            run_agent(agent)
    else:
        print("Invalid mode. Exiting.")

if __name__ == "__main__":
    main()
