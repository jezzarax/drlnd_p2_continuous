import json
import logging
import os
import sys

from unityagents import UnityEnvironment

from agents import *
from models import *

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout
)

ENVIRONMENT_BINARY = os.environ['DRLUD_P2_V2_ENV']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_prefix = "./hp_multi_agent_search_results/" if torch.cuda.is_available() else "./hp_single_agent_search_results/"


def train(agent, environment, n_episodes=1000, max_t=2000, store_weights_to="checkpoint.pth"):
    scores = []  # list containing scores from each episode
    for i_episode in range(1, n_episodes + 1):
        env_info = environment.reset(train_mode=True)[agent.name]
        states = env_info.vector_observations
        episode_scores = []
        for t in range(max_t):
            actions = agent.act(states)
            env_info = environment.step(actions)[agent.name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            episode_scores.append(sum(rewards)/len(rewards))
            if all(dones):
                break

        scores.append(sum(episode_scores))
        last_100_steps_mean = np.mean(scores[-100:])
        print('\rEpisode {}\tAverage Score: {:.2f}\tLast score: {:.2f}'.format(i_episode,
                                                                               last_100_steps_mean,
                                                                               np.mean(scores[-1])), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tLast score: {:.2f}'.format(i_episode,
                                                                                   last_100_steps_mean,
                                                                                   np.mean(scores[-1])))

        torch.save(agent.actor_local.state_dict(),
                   store_weights_to.replace("eps", str(n_episodes)).replace("role", "actor"))
        torch.save(agent.critic_local.state_dict(),
                   store_weights_to.replace("eps", str(n_episodes)).replace("role", "critic"))
    return scores


def prepare_environment():
    return UnityEnvironment(file_name=ENVIRONMENT_BINARY)


def infer_environment_properties(environment):
    brain_name = environment.brain_names[0]
    brain = environment.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = environment.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]
    return brain_name, num_agents, action_size, state_size


def prepare_agent(agent_config: ac_parm, seed):
    return Agent(agent_config, device, seed)


algorithm_factories = {
    "ddpg": prepare_agent
}

simulation_hyperparameter_reference = {
    1:  ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    2:  ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, False, False, 1),
    10: ac_parm(-1, -1, int(1e5), "", 128,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    11: ac_parm(-1, -1, int(1e5), "", 1024, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    12: ac_parm(-1, -1, int(1e5), "", 4096, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    13: ac_parm(-1, -1, int(1e5), "", 512,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    14: ac_parm(-1, -1, int(1e5), "", 384,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    20: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 2e-4, 0,    1, False, False, 1),
    21: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 4e-4, 0,    1, False, False, 1),
    22: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 5e-5, 0,    1, False, False, 1),
    23: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 8e-4, 0,    1, False, False, 1),
    24: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-5, 0,    1, False, False, 1),
    25: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 2e-5, 0,    1, False, False, 1),
    30: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 2e-4, 1e-4, 0,    1, False, False, 1),
    31: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 4e-4, 1e-4, 0,    1, False, False, 1),
    32: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 7e-5, 1e-4, 0,    1, False, False, 1),
    33: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 5e-5, 1e-4, 0,    1, False, False, 1),
    34: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-5, 1e-4, 0,    1, False, False, 1),
    35: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 5e-5, 5e-5, 0,    1, False, False, 1),
    36: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-5, 1e-5, 0,    1, False, False, 1),
    40: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 5e-3, 1, False, False, 1),
    41: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 1e-2, 1, False, False, 1),
    42: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 1e-3, 1, False, False, 1),
    43: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 5e-4, 1, False, False, 1),

    # Large sized memory to check for stability effect
    50: ac_parm(-1, -1, int(1e6), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    51: ac_parm(-1, -1, int(1e6), "", 1024, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    52: ac_parm(-1, -1, int(1e7), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    53: ac_parm(-1, -1, int(1e7), "", 1024, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    55: ac_parm(-1, -1, int(1e8), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    56: ac_parm(-1, -1, int(1e8), "", 1024, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),
    57: ac_parm(-1, -1, int(1e8), "", 4096, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 1),

    # Check the effect of gradient clipping on the overall stability
    60: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, True, 1),
    61: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, False, True, 1),
    62: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1,  True, True, 1),
    63: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1,  True, True, 1),

    # Check the effect of decreasing of the density of learning sessions over time
    70: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False,  2),
    71: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, False, False,  2),
    72: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1,  True, False,  2),
    73: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1,  True, False,  2),
    74: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False,  5),
    75: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, False, False,  5),
    76: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1,  True, False,  5),
    77: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1,  True, False,  5),
    78: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, 20),
    79: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, False, False, 20),
    80: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1,  True, False, 20),
    81: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1,  True, False, 20),

    # Check for general stability of outcomes over multiple runs with various random seed
    90: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,   25, False, False, 1),
    91: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,   25, False, False, 1),
    92: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,   25,  True, False, 1),
    93: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,   25,  True, False, 1),
}


def run_training_session(agent_factory, agent_config: ac_parm, id):
    env = prepare_environment()
    (brain_name, num_agents, action_size, state_size) = infer_environment_properties(env)
    agent_config = agent_config._replace(action_size=action_size, state_size=state_size, name=brain_name)

    scores = []
    for seed in range(agent_config.times):
        agent = agent_factory(agent_config, seed)
        scores.append(
            train(agent, env, store_weights_to=f"{path_prefix}set{id}_weights_role_episode_eps_seed_{seed}.pth"))
    env.close()
    return scores


def ensure_training_run(id: int, parm: ac_parm):
    if os.path.isfile(f"{path_prefix}set{id}_results.json"):
        logging.info(f"Skipping configuration {id} with following parameters {parm}")
    else:
        logging.info(f"Running {id} with following parameters {parm}")
        run_result = run_training_session(algorithm_factories["ddpg"], parm, id)
        with open(f"{path_prefix}set{id}_results.json", "w") as fp:
            json.dump(run_result, fp)


if __name__ == "__main__":
    for parm_id in simulation_hyperparameter_reference:
        ensure_training_run(parm_id, simulation_hyperparameter_reference[parm_id])
