#!/usr/bin/env python3.9
import os,sys
import argparse
import time
from callbacks import d_done_callback, d_info_callback
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_deception.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=scenario.info_callback, done_callback=scenario.done_callback, shared_viewer = False)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=scenario.info_callback, done_callback=scenario.done_callback, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        done_n = [0,0]
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, info_n = env.step(act_n)
        #print("DONE: " + str(done_n[0]) + '\t' + str(done_n[1]))
        #print("OBS: " + str(obs_n[0]) + '\n\t' + str(obs_n[1]))
        print("REWARD: " + str(reward_n[0]) + '\t' + str(reward_n[1]))
        # print("INFO: " + str(info_n["n"][1]))
        # render all agent views
        env.render()
        time.sleep(0.05)
        if (done_n[0] or done_n[1]):
            env.reset()
        # display rewards
        # for agent in env.world.agents:
        #     print(agent.name + " reward: %0.3f" % env._get_reward(agent))
