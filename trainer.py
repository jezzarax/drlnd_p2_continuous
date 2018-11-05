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

launch_parm = namedtuple("launch_parm", ["algorithm", "times", "hparm"])
env_parm = namedtuple("env_parm", ["state_size", "action_size", "brain_name", "agents_num"])


ENVIRONMENT_BINARY = os.environ['DRLUD_P2_V2_ENV']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_prefix = "./hp_multi_agent_search_results/" if torch.cuda.is_available() else "./hp_single_agent_search_results/"


def train(agent, environment, n_episodes=500, max_t=2000, store_weights_to="checkpoint.pth"):
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
            if any(dones):
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



def prepare_ddpg_agent(agent_config: ddpg_parm, env_parm: env_parm, seed):
    return DDPGAgent(agent_config, env_parm, device, seed)


algorithm_factories = {
    "ddpg": prepare_ddpg_agent
}

simulation_hyperparameter_reference = {
    # Check the effect of gradient clipping on the overall stability
    60: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 64,   0.99, 1e-3, 1e-4, 1e-4, 1e-2, False,  True, 1)),
    61: launch_parm("ddpg",  1, ddpg_parm(int(1e6), 512,  0.99, 1e-3, 1e-5, 1e-4, 1e-2, False,  True, 2)),
    62: launch_parm("ddpg",  1, ddpg_parm(int(1e6), 512,  0.99, 1e-3, 1e-5, 1e-4, 0,    False,  True, 4)),
#    63: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-3, 0,     True,  True, 1)),

    # Check the effect of decreasing of the density of learning sessions over time
    70: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    False, False,  2)),
    71: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    False, False,  2)),
    72: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-4, 0,     True, False,  2)),
    73: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-3, 0,     True, False,  2)),
    74: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    False, False,  5)),
    75: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    False, False,  5)),
    76: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-4, 0,     True, False,  5)),
    77: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-3, 0,     True, False,  5)),
    78: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    False, False, 20)),
    79: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    False, False, 20)),
    80: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-4, 0,     True, False, 20)),
    81: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-3, 0,     True, False, 20)),

    # Check for general stability of outcomes over multiple runs with various random seed
    90: launch_parm("ddpg", 25, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    False, False, 1)),
    91: launch_parm("ddpg", 25, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    False, False, 1)),
    92: launch_parm("ddpg", 25, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-4, 0,     True, False, 1)),
    93: launch_parm("ddpg", 25, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-3, 0,     True, False, 1)),
}


def run_training_session(agent_factory, run_config: launch_parm, id):
    env = prepare_environment()
    (brain_name, num_agents, action_size, state_size) = infer_environment_properties(env)
    agent_config = run_config.hparm
    scores = []
    for seed in range(run_config.times):
        agent = agent_factory(agent_config, env_parm(state_size, action_size, brain_name, num_agents), seed)
        scores.append(
            train(agent, env, store_weights_to=f"{path_prefix}set{id}_weights_role_episode_eps_seed_{seed}.pth"))
    env.close()
    return scores


def ensure_training_run(id: int, parm: launch_parm):
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
