import math
import random
import time
import pickle
import numpy as np
import pygame
import itertools
import matplotlib.pyplot as plt
from numba import jit


win = None


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

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
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


class Tri:
    def __init__(self, a, color):
        self.a = a
        self.color = color

    def get_vects(self):
        return [
            self.a,
        ]

    def draw(self):
        self.a.draw(self.color)


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
                if int(elt) == 0:
                    x1 = indexL * self.size
                    y1 = indexR * self.size
                    self.wall_points.append(
                        Rect(
                            Vect(x1, y1, x1 + self.size, y1),
                            Vect(x1, y1, x1, y1 + self.size),
                            Vect(x1 + self.size, y1 + self.size, x1 + self.size, y1),
                            Vect(x1 + self.size, y1 + self.size, x1, y1 + self.size),
                            (122, 122, 122)
                        ))
                if int(elt) == 2:
                    x1 = indexL * self.size
                    y1 = indexR * self.size
                    self.bonus_points.append(
                        Tri(
                            Vect(x1 + self.size / 5, y1 + self.size / 2, x1 + self.size - self.size / 5, y1 + self.size / 2),
                            (0, 255, 255)
                        )
                    )
                if int(elt) == 3:
                    x1 = indexL * self.size
                    y1 = indexR * self.size
                    self.bonus_points.append(
                        Tri(
                            Vect(x1 + self.size / 2, y1 + self.size / 5, x1 + self.size / 2, y1 + self.size - self.size / 5),
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
        self.impacts = [0 for i in range(4)]

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
            #Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY + self.sensor_size),
            #Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY - self.sensor_size),
            #Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY - self.sensor_size),
            #Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX, self.centerY - self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY),
            Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY),
        ]

    def update(self, x, y):
        Lreward = -1
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
            #Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY + self.sensor_size),
            #Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY - self.sensor_size),
            #Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY - self.sensor_size),
            #Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY + self.sensor_size),
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
                        Lreward = -300
                        collide = True
                        break
                if collide:
                    break
            if collide:
                break

        if Lreward != -300:
            for bonus in self.map.bonus_points:
                collide = False
                for ib, b in enumerate(bonus.get_vects()):
                    for vectC in self.car.get_vects():
                        res = intersectLines((vectC.x1, vectC.y1), (vectC.x2, vectC.y2), (b.x1, b.y1),
                                             (b.x2, b.y2))
                        if res[2]:
                            Lreward = 20
                            collide = True
                            self.map.collected_bonus.append(ib)
                            break
                    if collide:
                        break
                if collide:
                    break

        for idx, sensor in enumerate(self.sensors):
            collide = False
            for wall in self.map.wall_points:
                for vect in wall.get_vects():
                    res = intersectLines((sensor.x1, sensor.y1), (sensor.x2, sensor.y2), (vect.x1, vect.y1),
                                         (vect.x2, vect.y2))
                    if res[2]:
                        collide = True
                        x, y = int(res[0]), int(res[1])
                        dist = distance((self.centerX, self.centerY), (x, y)) - self.size / 2
                        self.impacts.append(-max(1, min(round(dist / 5), 4)))
                        minCoord = (x, y)
                        self.points.append(
                            minCoord
                        )
                        break
                if collide:
                    break

            if not collide:
                for bonus in self.map.bonus_points:
                    for b in bonus.get_vects():
                        res = intersectLines((sensor.x1, sensor.y1), (sensor.x2, sensor.y2), (b.x1, b.y1),
                                             (b.x2, b.y2))
                        if res[2]:
                            collide = True
                            x, y = int(res[0]), int(res[1])
                            dist = distance((self.centerX, self.centerY), (x, y)) - self.size / 2
                            self.impacts.append(max(1, min(round(dist / 5), 4)))
                            minCoord = (x, y)
                            self.points.append(
                                minCoord
                            )
                            break
                    if collide:
                        break

            if not collide:
                self.impacts.append(-1)

        return Lreward, tuple(self.impacts)

    def draw(self):
        for sensor in self.sensors:
            sensor.draw((0, 255, 0))
        for p in self.points:
            pygame.draw.circle(win, (255, 0, 255), p, 1)
        self.car.draw()

    def get_state(self):
        return tuple(self.impacts)


start = time.time()
q_table = {}
for i in itertools.product([-4, -3, -2, -1, 0, 1, 2, 3, 4], repeat=4):
    q_table[i] = [random.randint(0, 4) for i in range(4)]
print(time.time() - start)

HM_EPISODES = 1000
MOVE_PENALTY = 1  # feel free to tinker with these!
ENEMY_PENALTY = 300  # feel free to tinker with these!

FOOD_REWARD = 25  # feel free to tinker with these!
TARGET_REWARD = 100
epsilon = 0.3  # randomness
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 100  # how often to play through env visually.

LEARNING_RATE = 0.5
DISCOUNT = 0.95
VEL = 10

MAP_SIZE = 50

pygame.init()
win = pygame.display.set_mode((500, 500))

episode_rewards = []
for episode in range(HM_EPISODES):
    map = Map('map.txt', MAP_SIZE, win)
    p = Player(15 * 5, 15 * 5, 15, map, sensor_size=50)
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False
    r = False
    while 1:
    #for i in range(200):
        obs = p.get_state()
        # pygame.time.delay(5000)


        if np.random.random() > epsilon:
            r = True
            action = np.argmax(q_table[obs])
        else:
            r = False
            action = np.random.randint(0, 4)

        #action = np.argmax(q_table[obs])
        #action = np.argmax((q_table[obs]))

        x = 0
        y = 0


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



        """
        if action == 0:
            y -= VEL

        if action == 1:
            x += VEL

        if action == 2:
            y += VEL

        if action == 3:
            x -= VEL
        """
        reward, new_obs = p.update(x, y)
        print(reward, new_obs)
        max_future_q = np.argmax(q_table[new_obs])  # max Q value for this new obs
        current_q = q_table[obs][action]  # current Q for our chosen action
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        if reward == -300:
            new_q = -300

        q_table[obs][action] = new_q

        episode_reward += reward
        if reward == -300:
            if r:
                print("random ", end='')
            print("DIED", obs, action)
            break

        if show:
            #pygame.time.delay(1000)
            #print(obs)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            win.fill((0, 0, 0))
            map.draw()
            p.draw()
            pygame.display.update()

    #print(f"{episode} - {episode_reward}")
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

        # p.update(x, y)

#moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in episode_rewards])
plt.ylabel(f"Rewards")
plt.xlabel("episode #")
plt.show()

with open(f"qtable.pickle", "wb") as f:
    pickle.dump(q_table, f)


pygame.quit()
