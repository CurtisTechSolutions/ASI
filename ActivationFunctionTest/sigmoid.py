import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

env = gym.make("CartPole-v1")

state_size = int(env.observation_space.shape[0])
action_size = int(env.action_space.n)

print("State Size:", state_size)
print("Action Size:", action_size)

model = Sequential([
    Dense(24, activation="sigmoid", input_shape=(state_size,)),
    Dense(action_size, activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse")

gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
episodes = 250


for episode in range(episodes):

    state, _ = env.reset()
    state = state.reshape(1, state_size)

    done = False

    while not done:

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state, verbose=0))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = next_state.reshape(1, state_size)

        target = reward
        if not done:
            target += gamma * np.max(model.predict(next_state, verbose=0))

        q_values = model.predict(state, verbose=0)
        q_values[0][action] = target

        model.fit(state, q_values, epochs=1, verbose=0)
        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode+1} completed")



state, _ = env.reset()
state = state.reshape(1, state_size)

done = False
total_reward = 0

while not done:
    action = np.argmax(model.predict(state, verbose=0))

    next_state, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward
    state = next_state.reshape(1, state_size)
    done = terminated or truncated

print("Total Reward:", total_reward)