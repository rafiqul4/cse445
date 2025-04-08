import numpy as np
import random
import pickle
import os

# Tic-Tac-Toe Environment
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 = AI (X), -1 = Human (O)
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def get_valid_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            reward, done = self.check_winner()
            self.current_player *= -1
            return self.get_state(), reward, done
        return self.get_state(), -10, False  # Invalid move

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return (1, True) if self.current_player == 1 else (-1, True)
        if abs(self.board.trace()) == 3 or abs(np.fliplr(self.board).trace()) == 3:
            return (1, True) if self.current_player == 1 else (-1, True)
        if not self.get_valid_moves():
            return 0, True  # Draw
        return 0, False

    def print_board(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print(" | ".join(symbols[cell] for cell in row))
            print("-" * 9)

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay=0.99):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state, valid_moves):
        max_next_q = max([self.get_q_value(next_state, a) for a in valid_moves], default=0)
        self.q_table[(state, action)] = (1 - self.alpha) * self.get_q_value(state, action) + \
                                        self.alpha * (reward + self.gamma * max_next_q)

    def choose_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        q_values = [self.get_q_value(state, a) for a in valid_moves]
        return valid_moves[np.argmax(q_values)]

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

# Quick training for AI
def train_agent(episodes=10000):
    env = TicTacToe()
    agent = QLearningAgent()

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            valid_moves = env.get_valid_moves()
            action = agent.choose_action(state, valid_moves)
            next_state, reward, done = env.make_move(*action)
            agent.update_q_value(state, action, reward, next_state, env.get_valid_moves())
            state = next_state
        agent.decay_epsilon()

    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

# Play against trained AI
def play_against_ai():
    if not os.path.exists("q_table.pkl"):
        train_agent()

    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    env = TicTacToe()
    agent = QLearningAgent()
    agent.q_table = q_table
    env.reset()

    print("You are 'X' and AI is '0'. Enter your move as: row col (0-2)")
    env.print_board()

    while True:
        # Human move
        try:
            row, col = map(int, input("Your move (row col): ").split())
            if env.board[row][col] != 0:
                print("❌ Invalid move! Try again.")
                continue
        except:
            print("⚠️ Invalid input! Enter two numbers like: 1 2")
            continue

        _, reward, done = env.make_move(row, col)
        env.print_board()

        if done:
            if reward == -1:
                print("🎉 You win!")
            elif reward == 0:
                print("🤝 It's a draw!")
            break

        # AI move
        state = env.get_state()
        ai_action = agent.choose_action(state, env.get_valid_moves())
        _, reward, done = env.make_move(*ai_action)
        print(f"AI moves to: {ai_action}")
        env.print_board()

        if done:
            if reward == 1:
                print("😞 AI wins!")
            elif reward == 0:
                print("🤝 It's a draw!")
            break

# Run game
if __name__ == "__main__":
    play_against_ai()
