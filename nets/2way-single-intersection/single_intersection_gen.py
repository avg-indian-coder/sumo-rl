# Taken and modified from https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control

import numpy as np
import random
import math
import argparse
import os
import sys

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed, filename):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open(filename, "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="w_t t_n"/>
            <route id="W_E" edges="w_t t_e"/>
            <route id="W_S" edges="w_t t_s"/>
            <route id="N_W" edges="n_t t_w"/>
            <route id="N_E" edges="n_t t_e"/>
            <route id="N_S" edges="n_t t_s"/>
            <route id="E_W" edges="e_t t_w"/>
            <route id="E_N" edges="e_t t_n"/>
            <route id="E_S" edges="e_t t_s"/>
            <route id="S_W" edges="s_t t_w"/>
            <route id="S_N" edges="s_t t_n"/>
            <route id="S_E" edges="s_t t_e"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()  # samples from uniform dist [0, 1)
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)



# prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#                                   description="""Generating Routes""")
# prs.add_argument("-test", action="store_true", default=False, help="training or testing.\n")
# #prs.add_argument("-secs", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
# args = prs.parse_args()

# #numsecs = args.secs
# if (args.test):
#     numsecs = 5400
#     filename = "network/test_routes.rou.xml"
# else:
#     numsecs = 10000
#     filename = "network/final_routes.rou.xml"

train_uniform = "train/uniform.rou.xml"
train_ns = "train/ns_loaded.rou.xml"
train_ew = "train/ew_loaded.rou.xml"

test_uniform = "test/uniform.rou.xml"
test_ns = "test/ns_loaded.rou.xml"
test_ew = "test/ew_loaded.rou.xml"

TrafficGen = TrafficGenerator(5400, 1000)

TrafficGen.generate_routefile(0, train_uniform)
TrafficGen.generate_routefile(1, train_ns)
TrafficGen.generate_routefile(2, train_ew)

TrafficGen.generate_routefile(30, test_uniform)
TrafficGen.generate_routefile(31, test_ns)
TrafficGen.generate_routefile(32, test_ew)


