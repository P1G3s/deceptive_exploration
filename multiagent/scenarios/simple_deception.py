import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        # dim_c -> communication channel dimensionality
        world.dim_c = 0
        num_agents = 2
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = 3 #num_agents - 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.max_speed = 1 if i < num_adversaries else 2
            agent.size = 0.10
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.steps_count = 0
        world.agents[0].color = np.array([0.85, 0.35, 0.35])    # adversary
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            #agent.state.p_pos = np.random.uniform(-2.5, +2.5, world.dim_p)
            agent.state.p_pos = np.zeros(world.dim_p)
            if (agent.adversary):   # spawn defender on the upper half
                agent.state.p_pos[0] = random.uniform(-3.0, +3.0)
                agent.state.p_pos[1] = random.uniform(+0.5, +3.0)
            else:   # spawn invader on the bottom
                agent.state.p_pos[0] = random.uniform(-3.0, +3.0)
                agent.state.p_pos[1] = random.uniform(-3.0, -0.5)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):  # spawn landmarks on the upper half
            #landmark.state.p_pos = np.random.uniform(-2.5, +2.5, world.dim_p)
            landmark.state.p_pos = np.zeros(world.dim_p)
            landmark.state.p_pos[0] = random.uniform(-3.0, +3.0)
            landmark.state.p_pos[1] = random.uniform(+0.5, +3.0)
            landmark.state.p_vel = np.zeros(world.dim_p)
        adversary_agent = self.adversaries(world)[0]
        friendly_agent = self.friendly_agents(world)[0]
        world.last_f2g_dist = np.sqrt(np.sum(np.square(friendly_agent.state.p_pos - friendly_agent.goal_a.state.p_pos)))
        world.last_f2a_dist = np.sqrt(np.sum(np.square(adversary_agent.state.p_pos - friendly_agent.state.p_pos)))

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def friendly_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Rewarded based on how close any friendly agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True
        steps_penalty_ratio = 0.0005

        adversary_agent = self.adversaries(world)[0]
        weight = 0.01
        # distance between friendly and adversary
        f2a_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - adversary_agent.state.p_pos)))
        # distance between friendly and goal
        f2g_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
        # adv_rew = weight * (f2a_dist - world.last_f2a_dist)
        # pos_rew = -((1-weight) * (f2g_dist - world.last_f2g_dist))
        # rew = 20*(adv_rew + pos_rew)
        rew = 40*(-(f2g_dist - world.last_f2g_dist))
        # rew = 0.1*(-(f2g_dist))
        world.last_f2a_dist = f2a_dist
        world.last_f2g_dist = f2g_dist
        if (f2a_dist < 2*agent.size):
            rew -= 600
        if (f2g_dist < 3*agent.goal_a.size):
            rew += 600
        # rew -= steps_penalty_ratio * (world.steps_count^2)
        rew = round(rew,3)
            
        return rew

    def adversary_reward(self, agent, world):
        shaped_reward = True
        friendly_agent = self.friendly_agents(world)[0]

        f2a_dist = np.sqrt(np.sum((np.square(agent.state.p_pos - friendly_agent.state.p_pos))))
        f2g_dist = np.sqrt(np.sum(np.square(friendly_agent.state.p_pos - friendly_agent.goal_a.state.p_pos)))
        rew = 40*(-(f2a_dist - world.last_f2a_dist))
        # rew = 0.1*(-f2a_dist)
        if shaped_reward:  # distance-based reward
            if (f2a_dist < 2*friendly_agent.size):
                rew += 600
            elif (f2g_dist < 3*agent.goal_a.size):
                rew -= 600
        rew = round(rew,3)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        adversary_agent = self.adversaries(world)[0]
        friendly_agent = self.friendly_agents(world)[0]
        obs = None
        entity_pos = []
        world.steps_count += 1
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            obs = np.concatenate([friendly_agent.goal_a.state.p_pos - friendly_agent.state.p_pos] + \
                     [adversary_agent.state.p_pos - friendly_agent.state.p_pos] + \
                     entity_pos)
                     # [[world.steps_count]] + \
            return obs
        else:
            obs = np.concatenate([adversary_agent.state.p_pos - friendly_agent.state.p_pos] + \
                     entity_pos)
            return obs

    def done_callback(self, agent, world):
    # 0 -> adversary,   1 -> friendly
    # this callback get called by the adversary and friendly seperately
        friendly_agent = self.friendly_agents(world)[0]
        if (agent.adversary):
            # distance from adversary_agent to friendly_agent
            adversary_agent = [agent for agent in world.agents if agent.adversary][0]
            f2a_dist = np.sqrt(np.sum(np.square(adversary_agent.state.p_pos - friendly_agent.state.p_pos)))
            if (f2a_dist < 2*friendly_agent.size):
                return 1
        else:
            # distance from friendly_agent to true landmark
            f2g_dist = np.sqrt(np.sum(np.square(friendly_agent.state.p_pos - friendly_agent.goal_a.state.p_pos)))
            if (f2g_dist < 3*friendly_agent.goal_a.size):
                return 1
        return 0


    def info_callback(self, agent, world):
        # we dont need any info when this function is called by the adversary
        if (agent.adversary):
            return []
        else:
            adversary_agent = self.adversaries(world)[0]
            friendly_agent = self.friendly_agents(world)[0]
            fake_landmarks = [landmark for landmark in world.landmarks if (landmark != friendly_agent.goal_a)]
            goal = friendly_agent.goal_a
    
            goal_dist = np.sqrt(np.sum(np.square(friendly_agent.state.p_pos - goal.state.p_pos)))
            landmark_dists = [np.sqrt(np.sum(np.square(friendly_agent.state.p_pos - l.state.p_pos))) for l in fake_landmarks]
            landmark_dists.insert(0,goal_dist)
    
        return landmark_dists
        # distances from friendly_agent to all the landmarks


