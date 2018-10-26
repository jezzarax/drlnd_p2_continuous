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
path_prefix = "./hp_single_agent_search_results/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            episode_scores.append(sum(rewards))
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
    1:  ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    2:  ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, False),
    10: ac_parm(-1, -1, int(1e5), "", 128,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    11: ac_parm(-1, -1, int(1e5), "", 1024, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    12: ac_parm(-1, -1, int(1e5), "", 4096, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    13: ac_parm(-1, -1, int(1e5), "", 512,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    14: ac_parm(-1, -1, int(1e5), "", 384,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    20: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 2e-4, 0,    1, False),
    21: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 4e-4, 0,    1, False),
    22: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 5e-5, 0,    1, False),
    23: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 8e-4, 0,    1, False),
    24: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-5, 0,    1, False),
    25: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 2e-5, 0,    1, False),
    30: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 2e-4, 1e-4, 0,    1, False),
    31: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 4e-4, 1e-4, 0,    1, False),
    32: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 7e-5, 1e-4, 0,    1, False),
    33: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 5e-5, 1e-4, 0,    1, False),
    34: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-5, 1e-4, 0,    1, False),
    35: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 5e-5, 5e-5, 0,    1, False),
    36: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-5, 1e-5, 0,    1, False),
    40: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 5e-3, 1, False),
    41: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 1e-2, 1, False),
    42: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 1e-3, 1, False),
    43: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 5e-4, 1, False),

    50: ac_parm(-1, -1, int(1e6), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    51: ac_parm(-1, -1, int(1e6), "", 1024, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    52: ac_parm(-1, -1, int(1e7), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    53: ac_parm(-1, -1, int(1e7), "", 1024, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    54: ac_parm(-1, -1, int(1e8), "", 1024, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),
    55: ac_parm(-1, -1, int(1e8), "", 4096, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, False),

    101: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, True),
    102: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, True),
    110: ac_parm(-1, -1, int(1e5), "", 128,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, True),
    111: ac_parm(-1, -1, int(1e5), "", 1024, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, True),
    112: ac_parm(-1, -1, int(1e5), "", 4096, 0.99, 1e-3, 1e-4, 1e-4, 0,    1, True),
    113: ac_parm(-1, -1, int(1e5), "", 512,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, True),
    114: ac_parm(-1, -1, int(1e5), "", 384,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, True),
    120: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 2e-4, 0,    1, True),
    121: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 4e-4, 0,    1, True),
    122: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 5e-5, 0,    1, True),
    123: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 8e-4, 0,    1, True),
    124: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-5, 0,    1, True),
    125: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 2e-5, 0,    1, True),
    130: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 2e-4, 1e-4, 0,    1, True),
    131: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 4e-4, 1e-4, 0,    1, True),
    132: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 7e-5, 1e-4, 0,    1, True),
    133: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 5e-5, 1e-4, 0,    1, True),
    134: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-5, 1e-4, 0,    1, True),
    135: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 5e-5, 5e-5, 0,    1, True),
    136: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-5, 1e-5, 0,    1, True),
    140: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 5e-3, 1, True),
    141: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 1e-2, 1, True),
    142: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 1e-3, 1, True),
    143: ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 5e-4, 1, True),
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
