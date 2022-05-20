import numpy as np

def d_done_callback(agent, world):
    # this callback get called twice in one step, so we only check "done" when the function is called by the adversary for the sake of performance
    if (agent.adversary):
        rouge_agent = [agent for agent in world.agents if agent.adversary][0]
        good_agent = [agent for agent in world.agents if not agent.adversary][0]
        # distance from rouge_agent to good_agent
        r2g_dist = np.sqrt(np.sum(np.square(rouge_agent.state.p_pos - good_agent.state.p_pos)))
        if (r2g_dist < 2*good_agent.size):
            return 1

        # distance from godd_agent to true landmark
        g2l_dist = np.sqrt(np.sum(np.square(good_agent.state.p_pos - good_agent.goal_a.state.p_pos)))
        if (g2l_dist < 2*good_agent.goal_a.size):
            return 1

    return 0


def d_info_callback(agent, world):
    # we dont need any info when this function is called by the adversary
    if (agent.adversary):
        return []
    else:
        rouge_agent = [agent for agent in world.agents if agent.adversary][0]
        good_agent = [agent for agent in world.agents if not agent.adversary][0]
        fake_landmarks = [landmark for landmark in world.landmarks if (landmark != good_agent.goal_a)]
        true_landmark = good_agent.goal_a

        true_landmark_dist = np.sqrt(np.sum(np.square(good_agent.state.p_pos - true_landmark.state.p_pos)))
        landmark_dists = [np.sqrt(np.sum(np.square(good_agent.state.p_pos - l.state.p_pos))) for l in fake_landmarks]
        landmark_dists.insert(0,true_landmark_dist)

    return landmark_dists
    # distances from good_agent to all the landmarks

