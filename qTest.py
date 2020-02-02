import math
import random
import time
import numpy as np
import pygame
import matplotlib.pyplot as plt
import torch
import itertools


class Map:
    def __init__(self, txt_file, size, win):
        self.txt_file = txt_file
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()

        self.lines = lines
        self.size = size
        self.rects = []
        self.boost = []
        self.collected_bonus = []
        self.win = win

        self.final = None
        self.draw()

    def reset(self):
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()

        self.lines = lines
        self.rects = []
        self.boost = []
        self.collected_bonus = []

        self.final = None
        self.draw()

    def draw(self):
        self.rects = []
        self.boost = []
        bonus_index = 0
        for indexL, line in enumerate(self.lines):
            for indexR, elt in enumerate(line.split(',')):
                if int(elt) == 0:
                    tmp = pygame.draw.rect(
                        self.win,
                        (110, 110, 110),
                        (indexL * self.size, indexR * self.size, self.size, self.size))
                    self.rects.append(tmp)
                elif int(elt) == 100:
                    self.final = pygame.draw.rect(
                        self.win,
                        (0, 250, 0),
                        (indexL * self.size, indexR * self.size, self.size, self.size))
                elif int(elt) == 50:
                    tmp = None
                    if bonus_index not in self.collected_bonus:
                        tmp = pygame.draw.rect(
                            self.win,
                            (0, 110, 110),
                            (indexL * self.size, indexR * self.size, self.size, self.size))
                    self.boost.append(tmp)
                    bonus_index += 1

    def get_walls(self):
        return self.rects

    def get_bonus(self):
        return self.boost

    def collect_bonus(self, index):
        self.collected_bonus.append(index)

    def get_final(self):
        return self.final


class Player:
    def __init__(self, win, x, y, map, score):
        self.score = score

        self.last_max = 0
        self.win = win
        self.x = x
        self.y = y

        self.winner = False

        self.start_x = x
        self.start_y = y

        self.map = map
        self.circle = None
        self.alive = True
        self.sensors_config = [
            {'x': 0, 'y': -10, 'name': 'N'},
            {'x': 0, 'y': 10, 'name': 'S'},
            {'x': -10, 'y': 0, 'name': 'W'},
            {'x': 10, 'y': 0, 'name': 'E'},

            {'x': 0, 'y': -20, 'name': 'N'},
            {'x': 0, 'y': 20, 'name': 'S'},
            {'x': -20, 'y': 0, 'name': 'W'},
            {'x': 20, 'y': 0, 'name': 'E'},

            #{'x': -10, 'y': -10, 'name': 'NW'},
            #{'x': 10, 'y': 10, 'name': 'SE'},
            #{'x': -10, 'y': 10, 'name': 'SW'},
            #{'x': 10, 'y': -10, 'name': 'NE'},
        ]
        self.sensor_rect = []
        self.s_index_col = []
        self.s_index_colb = []
        self.draw()

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y

    def draw(self):
        self.circle = pygame.draw.circle(self.win, (0, 0, 255), (self.x, self.y), 5)
        self.sensor_rect = []

        for indexS, sensor in enumerate(self.sensors_config):
            color = (0, 0, 255)
            if indexS in self.s_index_col:
                color = (255, 0, 0)
            elif indexS in self.s_index_colb:
                color = (0, 255, 0)
            tmp = pygame.draw.line(self.win, color, (self.x, self.y), (self.x + sensor['x'], self.y + sensor['y']))
            self.sensor_rect.append({'rect': tmp, 'name': sensor['name']})

    def update(self):
        if self.circle is None:
            exit(1)
        walls = self.map.get_walls()
        for w in walls:
            if w.colliderect(self.circle):
                self.alive = False
                return -10000

        self.s_index_col = []
        for indexS, s in enumerate(self.sensor_rect):
            for w in walls:
                if s['rect'].colliderect(w):
                    self.s_index_col.append(indexS)

        bonuses = self.map.get_bonus()
        self.s_index_colb = []
        for indexS, s in enumerate(self.sensor_rect):
            for i, b in enumerate(bonuses):
                if b is not None and s['rect'].colliderect(b):
                    self.s_index_colb.append(indexS)

        for i, b in enumerate(bonuses):
            if b is not None and b.colliderect(self.circle):
                self.score += 25
                self.map.collect_bonus(i)
                return 1000

        return -1

    def is_alive(self):
        return self.alive

    def get_dist_from_start(self):
        return abs(math.sqrt((self.x - self.start_x) ** 2 + (self.y - self.start_y) ** 2))

    def get_state(self):
        return {
            "score": self.score,
            "x": self.x,
            "y": self.y,
            "win": self.win,
            "map": self.map
        }

    def get_initial_state(self):
        return [-1 for elt in range(0, len(self.sensors_config))]

    def get_sensor_state(self):
        elt = []
        for i in range(0, len(self.sensors_config)):
            if i in self.s_index_col:
                elt.append(1)
            elif i in self.s_index_colb:
                elt.append(2)
            else:
                elt.append(0)
        return tuple(elt)


SIZE = 50


def play(win):
    x = 100
    y = 100
    vel = 1

    map = Map('map.txt', 50, win)
    p = Player(win, x, y, map, 0)

    while p.is_alive() and p.winner is False:
        pygame.time.delay(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            p.x -= vel

        if keys[pygame.K_RIGHT]:
            p.x += vel

        if keys[pygame.K_UP]:
            p.y -= vel

        if keys[pygame.K_DOWN]:
            p.y += vel

        win.fill((0, 0, 0))

        p.update()
        map.draw()
        p.draw()
        pygame.display.update()

    pygame.quit()

pygame.init()
win = pygame.display.set_mode((500, 500))

#play(win)

HM_EPISODES = 100
MOVE_PENALTY = 1  # feel free to tinker with these!
ENEMY_PENALTY = 300  # feel free to tinker with these!

FOOD_REWARD = 25  # feel free to tinker with these!
TARGET_REWARD = 100
epsilon = 0.7  # randomness
EPS_DECAY = 0.7  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1  # how often to play through env visually.

LEARNING_RATE = 0.2
DISCOUNT = 0.95
VEL = 10

MAP_SIZE = 50


# play(win)

while 0:
    pygame.time.delay(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit(0)

    win.fill((255, 0, 0))
    pygame.draw.circle(win, (0, 0, 255), (20, 20), 10)
    pygame.display.update()

# 4 move
# 12 sensors

q_table = {}
for i in itertools.product([0, 1, 2], repeat=8):
    q_table[i] = [np.random.uniform(-100, 5) for i in range(4)]

print("q table generated")

episode_rewards = []
for episode in range(HM_EPISODES):
    map = Map('map.txt', MAP_SIZE, win)
    p = Player(win, 75, 75, map, 0)

    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = p.get_sensor_state()
        # print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            print("random")
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!

        if action == 0:
            p.x -= VEL

        if action == 1:
            p.x += VEL

        if action == 2:
            p.y -= VEL

        if action == 3:
            p.y += VEL

        if show:
            win.fill((1, 1, 1))

        # map.draw()
        # p.draw()

        reward = p.update()
        new_obs = p.get_sensor_state()
        max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs
        current_q = q_table[obs][action]  # current Q for our chosen action

        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            map.draw()
            p.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(0)

            pygame.display.update()

        episode_reward += reward
        if reward == -10000:
            break
        #if reward == -9:
        #    print("win")

    if show:
        print(f"{episode} - {episode_reward}")

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    np.pickle.dump(q_table, f)
