import math
import pygame
from numba import jit

winG = None

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
        pygame.draw.line(winG, color, (self.x1, self.y1), (self.x2, self.y2))


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
        global winG
        self.win = win
        winG = win
        self.txt_file = txt_file
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()

        self.lines = lines
        self.size = size

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
    def __init__(self, x, y, size, map, sensor_size, win, MOVE_PENALTY, DEATH_PENALTY, FOOD_REWARD):
        self.FOOD_REWARD = FOOD_REWARD
        self.DEATH_PENALTY = DEATH_PENALTY
        self.MOVE_PENALTY = MOVE_PENALTY
        self.win = win
        self.map = map
        self.sensor_size = sensor_size
        self.x = x
        self.y = y
        self.size = size
        self.points = []
        self.impacts = [0 for i in range(8)]

        self.centerX = self.x + size / 2
        self.centerY = self.y + size / 2
        self.refresh_car()
        self.refresh_sensors()
        self.update(0, 0)    # Trigger computation of initial state

    def refresh_car(self):
        self.car = Rect(
            Vect(self.x, self.y, self.x + self.size, self.y),
            Vect(self.x, self.y, self.x, self.y + self.size),
            Vect(self.x + self.size, self.y + self.size, self.x + self.size, self.y),
            Vect(self.x + self.size, self.y + self.size, self.x, self.y + self.size),
            (0, 0, 255)
        )
    def refresh_sensors(self):
        self.sensors = [
            Vect(self.centerX, self.centerY, self.centerX, self.centerY - self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY - self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY),
            Vect(self.centerX, self.centerY, self.centerX + self.sensor_size, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY + self.sensor_size),
            Vect(self.centerX, self.centerY, self.centerX -  self.sensor_size, self.centerY),
            Vect(self.centerX, self.centerY, self.centerX - self.sensor_size, self.centerY - self.sensor_size),
        ]
    def update(self, x, y):
        Lreward = -self.MOVE_PENALTY
        self.x = self.x + x
        self.y = self.y + y
        self.centerX = self.x + self.size / 2
        self.centerY = self.y + self.size / 2
        self.refresh_car()
        self.refresh_sensors()

        
        self.points = []
        self.impacts = []
        for wall in self.map.wall_points:
            collide = False
            for vectW in wall.get_vects():
                for vectC in self.car.get_vects():
                    res = intersectLines((vectC.x1, vectC.y1), (vectC.x2, vectC.y2), (vectW.x1, vectW.y1),
                                         (vectW.x2, vectW.y2))
                    if res[2]:
                        Lreward = -self.DEATH_PENALTY
                        collide = True
                        break
                if collide:
                    break
            if collide:
                break

        if Lreward != -self.DEATH_PENALTY:
            for indexB, bonus in enumerate(self.map.bonus_points):
                collide = False
                if indexB in self.map.collected_bonus:
                    continue
                for b in bonus.get_vects():
                    for vectC in self.car.get_vects():
                        res = intersectLines((vectC.x1, vectC.y1), (vectC.x2, vectC.y2), (b.x1, b.y1),
                                             (b.x2, b.y2))
                        if res[2]:
                            Lreward = self.FOOD_REWARD
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
                        colliders.append([dist, x, y, -1])

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
                        colliders.append([dist, x, y, 1])

            if collide:
                min_impact = colliders[0]
                for idx, impact in enumerate(colliders):
                    if min_impact[0] > impact[0]:
                        min_impact = impact

                sensor_dist = int(min_impact[0] - self.size / 2)
                self.impacts.append(min_impact[3] * sensor_dist)
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
            pygame.draw.circle(self.win, (255, 0, 255), p, 1)
        self.car.draw()

    def get_state(self):
        return tuple(self.impacts)
