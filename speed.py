import math
import random
import time
import pickle
import numpy as np
import pygame
import itertools
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib
import matplotlib.pyplot as plt
from numba import jit

win = None

HM_EPISODES = 5000
MOVE_PENALTY = 1  # feel free to tinker with these!
DEATH_PENALTY = 1000  # feel free to tinker with these!

FOOD_REWARD = 500  # feel free to tinker with these!
epsilon = 0.3  # randomness
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1000  # how often to play through env visually.

LEARNING_RATE = 0.3
DISCOUNT = 0.95
VEL = 5

MAP_SIZE = 50


# Sizing
# Move base is 1
# 1 block is 10 move
# sensor should see at 2 blocks
# init pos is 1.5 block in x and y
# allowed move is 5 * nb block in first row


@jit(nopython=True)
def get_orientation(p1, p2, p3):
    return ((p2[1] - p1[1]) * (p3[0] - p2[0])) - ((p3[1] - p2[1]) * (p2[0] - p1[0]))


@jit(nopython=True)
def intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    or1 = get_orientation((x1, y1), (x2, y2), (x4, y4)) * get_orientation((x1, y1), (x2, y2), (x3, y3))
    or2 = get_orientation((x3, y3), (x4, y4), (x1, y1)) * get_orientation((x3, y3), (x4, y4), (x2, y2))
    return or1 < 0 and or2 < 0


@jit(nopython=True)
def intersectLines(pt1, pt2, ptA, ptB):
    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0

    valid = intersect(x1, y1, x2, y2, x, y, xB, yB)

    return xi, yi, valid, r, s


@jit(nopython=True)
def distance(pt1, pt2):
    return math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))


class Vect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __str__(self):
        return f"{self.x1} {self.y1} {self.x2} {self.y2}"

    def draw(self, color):
        pygame.draw.line(win, color, (self.x1, self.y1), (self.x2, self.y2))


class Seg:
    def __init__(self, a, color):
        self.a = a
        self.color = color

    def get_vects(self):
        return [
            self.a,
        ]

    def draw(self):
        self.a.draw(self.color)


class Cross:
    def __init__(self, a, b, color):
        self.a = a
        self.b = b
        self.color = color

    def get_vects(self):
        return [
            self.a,
            self.b,
        ]

    def draw(self):
        self.a.draw(self.color)
        self.b.draw(self.color)


class Rect:
    def __init__(self, a, b, c, d, color):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.color = color

    def __str__(self):
        return f"{self.a}-{self.b} / {self.c}-{self.d}"

    def get_vects(self):
        return [
            self.a,
            self.b,
            self.c,
            self.d
        ]

    def draw(self):
        self.a.draw(self.color)
        self.b.draw(self.color)
        self.c.draw(self.color)
        self.d.draw(self.color)


class Map:
    def __init__(self, txt_file, size, win):
        self.txt_file = txt_file
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()

        self.lines = lines
        self.size = size
        self.win = win

        self.wall_points = []
        self.bonus_points = []
        self.collected_bonus = []
        self.process_points()

    def process_points(self):
        self.wall_points = []
        for indexL, line in enumerate(self.lines):
            for indexR, elt in enumerate(line.split(',')):
                x1 = indexL * self.size
                y1 = indexR * self.size
                if int(elt) == 1:
                    self.wall_points.append(
                        Seg(
                            Vect(x1 + self.size / 2, y1, x1 + self.size / 2, y1 + self.size),
                            (122, 122, 122)
                        ))
                if int(elt) == 2:
                    self.wall_points.append(
                        Seg(
                            Vect(x1, y1 + self.size / 2, x1 + self.size, y1 + self.size / 2),
                            (122, 122, 122)
                        ))
                ##########
                if int(elt) == 3:
                    self.wall_points.append(
                        Cross(
                            Vect(x1 + self.size / 2, y1 + self.size / 2, x1 + self.size, y1 + self.size / 2),
                            Vect(x1 + self.size / 2, y1 + self.size / 2, x1 + self.size / 2, y1 + self.size),
                            (122, 122, 122)
                        )
                    )
                if int(elt) == 4:
                    self.wall_points.append(
                        Cross(
                            Vect(x1 + self.size / 2, y1 + self.size / 2, x1 + self.size, y1 + self.size / 2),
                            Vect(x1 + self.size / 2, y1 + self.size / 2, x1 + self.size / 2, y1 - self.size),
                            (122, 122, 122)
                        )
                    )
                if int(elt) == 5:
                    self.wall_points.append(
                        Cross(
                            Vect(x1 + self.size / 2, y1 + self.size / 2, x1 - self.size, y1 + self.size / 2),
                            Vect(x1 + self.size / 2, y1 + self.size / 2, x1 + self.size / 2, y1 - self.size),
                            (122, 122, 122)
                        )
                    )
                if int(elt) == 6:
                    self.wall_points.append(
                        Cross(
                            Vect(x1 + self.size / 2, y1 + self.size / 2, x1 - self.size, y1 + self.size / 2),
                            Vect(x1 + self.size / 2, y1 + self.size / 2, x1 + self.size / 2, y1 + self.size),
                            (122, 122, 122)
                        )
                    )
                ##########
                if int(elt) == 7:
                    self.bonus_points.append(
                        Seg(
                            Vect(x1, y1 + self.size / 2, x1 + self.size, y1 + self.size / 2),
                            (0, 255, 255)
                        )
                    )
                if int(elt) == 8:
                    self.bonus_points.append(
                        Seg(
                            Vect(x1 + self.size / 2, y1, x1 + self.size / 2, y1 + self.size),
                            (0, 255, 255)
                        )
                    )

    def get_points(self):
        return self.wall_points

    def draw(self):
        for indexL, walls in enumerate(self.wall_points):
            walls.draw()
        for indexB, bonus in enumerate(self.bonus_points):
            if indexB not in self.collected_bonus:
                bonus.draw()


class Player:
    def __init__(self, x, y, size, map, sensor_size):
        self.map = map
        self.sensor_size = sensor_size
        self.x = x
        self.y = y
        self.size = size
        self.points = []
        self.impacts = [0 for i in range(8)]

        self.centerX = self.x + size / 2
        self.centerY = self.y + size / 2
        self.car = Rect(
            Vect(self.x, self.y, self.x + self.size, self.y),
            Vect(self.x, self.y, self.x, self.y + self.size),
            Vect(self.x + self.size, self.y + self.size, self.x + self.size, self.y),
            Vect(self.x + self.size, self.y + self.size, self.x, self.y + self.size),
            (0, 0, 255)
        )
        self.sensors = [
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY - self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY - self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX, self.centerY - self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY),
            Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY),
        ]

    def update(self, x, y):
        Lreward = -MOVE_PENALTY
        self.x = self.x + x
        self.y = self.y + y
        self.centerX = self.x + self.size / 2
        self.centerY = self.y + self.size / 2
        self.car = Rect(
            Vect(self.x, self.y, self.x + self.size, self.y),
            Vect(self.x, self.y, self.x, self.y + self.size),
            Vect(self.x + self.size, self.y + self.size, self.x + self.size, self.y),
            Vect(self.x + self.size, self.y + self.size, self.x, self.y + self.size),
            (255, 255, 255)
        )
        self.sensors = [
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY - self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY - self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX, self.centerY - self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY),
            Vect(self.centerX, self.centerY, self.centerX, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY),
        ]
        self.points = []
        self.impacts = []
        for wall in self.map.wall_points:
            collide = False
            for vectW in wall.get_vects():
                for vectC in self.car.get_vects():
                    res = intersectLines((vectC.x1, vectC.y1), (vectC.x2, vectC.y2), (vectW.x1, vectW.y1),
                                         (vectW.x2, vectW.y2))
                    if res[2]:
                        Lreward = -DEATH_PENALTY
                        collide = True
                        break
                if collide:
                    break
            if collide:
                break

        if Lreward != -DEATH_PENALTY:
            for indexB, bonus in enumerate(self.map.bonus_points):
                collide = False
                if indexB in self.map.collected_bonus:
                    continue
                for b in bonus.get_vects():
                    for vectC in self.car.get_vects():
                        res = intersectLines((vectC.x1, vectC.y1), (vectC.x2, vectC.y2), (b.x1, b.y1),
                                             (b.x2, b.y2))
                        if res[2]:
                            Lreward = FOOD_REWARD
                            collide = True
                            self.map.collected_bonus.append(indexB)
                            break
                    if collide:
                        break
                if collide:
                    break

        for idx, sensor in enumerate(self.sensors):
            collide = False
            colliders = []
            for wall in self.map.wall_points:
                for vect in wall.get_vects():
                    res = intersectLines((sensor.x1, sensor.y1), (sensor.x2, sensor.y2), (vect.x1, vect.y1),
                                         (vect.x2, vect.y2))
                    if res[2]:
                        collide = True
                        x, y = int(res[0]), int(res[1])
                        dist = distance((self.centerX, self.centerY), (x, y)) - self.size / 2
                        colliders.append([dist, x, y])
                        minCoord = (x, y)
                        #self.points.append(
                        #    minCoord
                        #)

            for indexB, bonus in enumerate(self.map.bonus_points):
                if indexB in self.map.collected_bonus:
                    continue
                for b in bonus.get_vects():
                    res = intersectLines((sensor.x1, sensor.y1), (sensor.x2, sensor.y2), (b.x1, b.y1),
                                         (b.x2, b.y2))
                    if res[2]:
                        collide = True
                        x, y = int(res[0]), int(res[1])
                        dist = distance((self.centerX, self.centerY), (x, y)) - self.size / 2
                        colliders.append([dist, x, y])
                        # self.impacts.append(max(1, min(round(dist / 5), 4)))
                        minCoord = (x, y)
                        #self.points.append(
                        #    minCoord
                        #)

            if collide:
                min_impact = colliders[0]
                for idx, impact in enumerate(colliders):
                    if min_impact[0] > impact[0]:
                        min_impact = impact

                self.impacts.append(-max(1, min(round(min_impact[0] / 5), 4)))
                self.points.append(
                    (min_impact[1], min_impact[2])
                )
            else:
                self.impacts.append(0)

        return Lreward, tuple(self.impacts)

    def draw(self):
        for sensor in self.sensors:
            sensor.draw((0, 255, 0))
        for p in self.points:
            pygame.draw.circle(win, (255, 0, 255), p, 1)
        self.car.draw()

    def get_state(self):
        return tuple(self.impacts)


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

MANUAL = False

pool = ThreadPool(4)
for episode in range(HM_EPISODES):
    map = Map('map.txt', MAP_SIZE, win)
    p = Player(15 * 5, 15 * 5, 15, map, sensor_size=75)
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False
    r = False
    # while 1:
    for i in range(200):
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
            max_future_q = np.random.randint(0, 4)

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

        if reward == -DEATH_PENALTY:
            break

        if show:
            # pygame.time.delay(1000)
            # print(obs)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

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

with open(f"qtable.pickle", "wb") as f:
    pickle.dump(q_table, f)

pygame.quit()
