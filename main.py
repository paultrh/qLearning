import random
import time
import pickle
import numpy as np
import pygame
import itertools
import matplotlib.pyplot as plt

from core import Map, Player, Config

win = None

HM_EPISODES = 10000
MOVE_PENALTY = 1  # feel free to tinker with these!
DEATH_PENALTY = 1000  # feel free to tinker with these!

FOOD_REWARD = 500  # feel free to tinker with these!
epsilon = 0.5  # randomness
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1000  # how often to play through env visually.

LEARNING_RATE = 0.3
DISCOUNT = 0.95
VEL = 3

MAP_SIZE = 50

EPISODE_TIMEOUT = 500
MANUAL = False


# Sizing
# Move base is 1
# 1 block is 10 move
# sensor should see at 2 blocks
# init pos is 1.5 block in x and y
# allowed move is 5 * nb block in first row

def generate_qtable():
    return {}
    start = time.time()
    qtable = {}
    for i in itertools.product([-4, -3, -2, -1, 0, 1, 2, 3, 4], repeat=8):  # complexity is 9 ** 8
        qtable[i] = 0
    print(time.time() - start)
    return qtable


q_table = generate_qtable()

pygame.init()
win = pygame.display.set_mode((500, 500))

already_dead_ops = set()
start = time.time()
episode_rewards = []
random_liberty = []
len_qtable = []
died_table = []

# Singleton that holds global variables
c = Config(win, MOVE_PENALTY, DEATH_PENALTY, FOOD_REWARD)

start_pos = [
    (15 * 5, 15 * 5),
    (50 * 10 - 15 * 5, 50 * 10 - 15 * 5),
    (50 * 10 - 15 * 5, 15 * 5),
    (15 * 5, 50 * 10 - 15 * 5)
]

for episode in range(HM_EPISODES):
    DIED = False
    map = Map('map.txt', MAP_SIZE)
    p = Player(*random.choice(start_pos), 15, map, sensor_size=75)
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False
    r = False
    # while 1:
    for i in range(EPISODE_TIMEOUT):
        obs = p.get_state()
        # pygame.time.delay(500)

        action = 0
        if np.random.random() > epsilon:
            r = False
            list_action_score = q_table.get(obs, None)
            if list_action_score:
                action = np.argmax(list_action_score)
            else:
                action = np.random.randint(0, 4)

        else:
            r = True
            action = np.random.randint(0, 4)

        # action = np.argmax(q_table[obs])
        # action = np.argmax((q_table[obs]))

        x = 0
        y = 0

        if MANUAL:
            pygame.event.wait()
            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                x -= VEL

            if keys[pygame.K_RIGHT]:
                x += VEL

            if keys[pygame.K_UP]:
                y -= VEL

            if keys[pygame.K_DOWN]:
                y += VEL
        else:

            # Move Up
            if action == 0:
                y -= VEL

            # Move Right
            if action == 1:
                x += VEL

            # Move Down
            if action == 2:
                y += VEL

            # Move Left
            if action == 3:
                x -= VEL

        # print(f"on pos {p.centerX} {p.centerY} {obs}") # 122.5 377.5

        reward, new_obs = p.update(x, y)

        list_action_score_future = q_table.get(new_obs, None)
        max_future_q = 0
        if list_action_score_future:
            max_future_q = np.argmax(list_action_score_future)
        else:
            max_future_q = np.random.randint(0, 1)

        # max_future_q = np.argmax(q_table[new_obs])  # max Q value for this new obs

        q_table.setdefault(obs, [0, 0, 0, 0])
        current_q = q_table.get(obs)[action]  # current Q for our chosen action
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        if reward == -DEATH_PENALTY:
            if not r and obs + (action,) in already_dead_ops:
                print(f"DIED AGAIN ON {obs} doing a {action} on pos {p.centerX} {p.centerY}")
            already_dead_ops.add(obs + (action,))
            new_q = -DEATH_PENALTY

        q_table[obs][action] = new_q

        episode_reward += reward

        if MANUAL:
            print("score {}".format(episode_reward))
            print("{} {}".format(p.centerX, p.centerY))

        if reward == -DEATH_PENALTY:
            DIED = True
            break

        if show:
            # pygame.time.delay(1000)
            # print(obs)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # running = False
                    pygame.quit()
                    exit(0)

            win.fill((0, 0, 0))
            map.draw()
            p.draw()
            pygame.display.update()

    if episode % SHOW_EVERY == 0:
        print(f"{episode} - {episode_reward} in {time.time() - start}")
        start = time.time()

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    random_liberty.append(epsilon)
    len_qtable.append(len(q_table))
    died_table.append(0 if DIED and r else 1)

    # p.update(x, y)

# moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in episode_rewards])
plt.ylabel(f"Rewards")
plt.xlabel("episode #")
plt.show()

plt.plot([i for i in random_liberty])
plt.ylabel(f"Liberty")
plt.xlabel("episode #")
plt.show()

plt.plot([i for i in len_qtable])
plt.ylabel(f"Len q table")
plt.xlabel("episode #")
plt.show()

plt.plot([i for i in died_table])
plt.ylabel(f"Did we die because random 0=KO / 1=OK ")
plt.xlabel("episode #")
plt.show()

with open(f"qtable.pickle", "wb") as f:
    pickle.dump(q_table, f)

pygame.quit()
