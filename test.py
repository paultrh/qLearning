import math
import random

import pygame
import torch

pygame.init()

win = pygame.display.set_mode((500, 500))


class Map:
    def __init__(self, txt_file, size):
        self.txt_file = txt_file
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()

        self.lines = lines
        self.size = size
        self.rects = []
        self.boost = []
        self.collected_bonus = []

        self.final = None

    def reset(self):
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()

        self.lines = lines
        self.rects = []
        self.boost = []
        self.collected_bonus = []

        self.final = None

    def draw(self):
        self.rects = []
        self.boost = []
        bonus_index = 0
        for indexL, line in enumerate(self.lines):
            for indexR, elt in enumerate(line.split(',')):
                if int(elt) == 0:
                    tmp = pygame.draw.rect(
                        win,
                        (110, 110, 110),
                        (indexL * self.size, indexR * self.size, self.size, self.size))
                    self.rects.append(tmp)
                elif int(elt) == 100:
                    self.final = pygame.draw.rect(
                        win,
                        (0, 250, 0),
                        (indexL * self.size, indexR * self.size, self.size, self.size))
                elif int(elt) == 50:
                    tmp = None
                    if bonus_index not in self.collected_bonus:
                        tmp = pygame.draw.rect(
                            win,
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
            {'x': 0, 'y': -30, 'name': 'N'},
            {'x': 0, 'y': 30, 'name': 'S'},
            {'x': -30, 'y': 0, 'name': 'W'},
            {'x': 30, 'y': 0, 'name': 'E'},

            {'x': -15, 'y': -15, 'name': 'NW'},
            {'x': 15, 'y': 15, 'name': 'SE'},
            {'x': -15, 'y': 15, 'name': 'SW'},
            {'x': 15, 'y': -15, 'name': 'NE'},

            {'x': -30, 'y': -60, 'name': 'NW'},
            {'x': 30, 'y': 60, 'name': 'SE'},
            {'x': -30, 'y': 60, 'name': 'SW'},
            {'x': 30, 'y': -60, 'name': 'NE'}
        ]
        self.sensor_rect = []
        self.s_index_col = []

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y

    def draw(self):
        self.circle = pygame.draw.circle(self.win, (0, 0, 255), (self.x, self.y), 10)

        self.s_index_col = []
        walls = self.map.get_walls()
        for indexS, s in enumerate(self.sensor_rect):
            for w in walls:
                if s['rect'].colliderect(w):
                    self.s_index_col.append(indexS)
        self.sensor_rect = []

        for indexS, sensor in enumerate(self.sensors_config):
            color = (0, 0, 255)
            if indexS in self.s_index_col:
                color = (255, 0, 0)
            tmp = pygame.draw.line(self.win, color, (self.x, self.y), (self.x + sensor['x'], self.y + sensor['y']))
            self.sensor_rect.append({'rect': tmp, 'name': sensor['name']})

    def update(self):
        if self.circle:
            walls = self.map.get_walls()
            for w in walls:
                if w.colliderect(self.circle):
                    self.alive = False

            final = self.map.get_final()
            if final.colliderect(self.circle):
                self.score += 10000
                self.winner = True

            bonuses = self.map.get_bonus()
            for i, b in enumerate(bonuses):
                if b is not None and b.colliderect(self.circle):
                    self.score += 500
                    self.map.collect_bonus(i)
                    return 500

        new_dist_from_start = self.get_dist_from_start()
        if new_dist_from_start > self.last_max:
            self.score += 1
            self.last_max = new_dist_from_start
            return 1

        return -1
        print("Score {}".format(self.score))

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
            else:
                elt.append(0)
        return elt


def play():
    x = 0
    y = 0
    vel = 5

    map = Map('map.txt', 50)
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


def play_states(states):
    x = 0
    y = 0
    vel = 5

    map = Map('map.txt', 50)
    p = Player(win, x, y, map, 0)

    count = 0
    while p.is_alive() and p.winner is False:
        pygame.time.delay(600)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)

        keys = pygame.key.get_pressed()

        if states[count] == "LEFT":
            p.x -= vel

        if states[count] == "RIGHT":
            p.x += vel

        if states[count] == "TOP":
            p.y -= vel

        if states[count] == "BOT":
            p.y += vel

        win.fill((0, 0, 0))

        p.update()
        map.draw()
        p.draw()
        pygame.display.update()
        count += 1

    pygame.quit()

'''
def get_next_rd_state():
    return random.choice(["BOT", "BOT", "RIGHT", "RIGHT"])


states = []
for i in range(1, 1000):
    states.append(get_next_rd_state())

play_states(states)
'''

class Game:

    def __init__(self):
        self.vel = 5
        x = 0
        y = 0
        self.map = Map('map.txt', 50)
        self.p = Player(win, x, y, self.map, 0)

    def reset(self):
        self.vel = 5
        x = 0
        y = 0
        self.map = Map('map.txt', 50)
        self.p = Player(win, x, y, self.map, 0)

    def get_initial_state(self):
        return self.p.get_initial_state()

    def get_state(self):
        return self.p.get_sensor_state()

    def step(self, action):
        if action == 0:
            self.p.x -= self.vel

        if action == 1:
            self.p.x += self.vel

        if action == 2:
            self.p.y -= self.vel

        if action == 3:
            self.p.y += self.vel

        self.map.draw()
        self.p.draw()
        r = self.p.update()

        return 0, r, self.p.winner, 0


GAME = Game()
print(torch.FloatTensor(GAME.get_initial_state()))
print(torch.FloatTensor(GAME.get_state()))
print(torch.FloatTensor(GAME.get_initial_state()) - torch.FloatTensor(GAME.get_state()))

print(GAME.step("RIGHT"))
print(GAME.step("RIGHT"))
print(GAME.step("RIGHT"))
print(GAME.step("LEFT"))

