import argparse
import logging
import random
import sys

import envs  # NOQA
import gym
import gym.wrappers
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments, explorers, replay_buffers, utils
from pfrl.agents.ddpg import DDPG
from pfrl.nn import BoundByTanh, ConcatObsAndAction
from pfrl.policies import DeterministicHead


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default="stir-v0",
        # default="stir_gazebo-v0",
        # default="Hopper-v4",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument(
        "--specialization",
        type=str,
        # default="StirEnvXYZPositionIngredients8MoveTool",
        # default="StirEnvXYZPositionIngredients8Stir",
        # default="StirEnvXYZPositionIngredients8KeepMovingIngredients",
        # default="StirEnvXYZVelocityIngredient4",
        default="StirEnvXYZPositionIngredients8StirWithMovingIngredients",
        help="StirEnv...",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="stir-ingredients8_toolxyz",
        help="xml file name",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed [0, 2 ** 32)"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40**6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10000,
        help="Minimum replay buffer size before "
        + "performing gradient updates.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Minibatch size"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Wrap env with gym.wrappers.Monitor.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=logging.INFO,
        help="Level of the root logger.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv
    )
    print("Output files are saved in {}".format(args.outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    def make_env(test):
        env = gym.make(
            args.env,
            specialization=args.specialization,
            xml=args.xml,
            render_mode="human",
        )
        # Unwrap TimeLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        env_seed = 2**32 - 1 - args.seed if test else args.seed
        # env.seed(env_seed)
        random.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        # if args.monitor:
        # env = pfrl.wrappers.Monitor(env, args.outdir)
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    q_func = nn.Sequential(
        ConcatObsAndAction(),
        nn.Linear(obs_size + action_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, 1),
    )
    policy = nn.Sequential(
        nn.Linear(obs_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, action_size),
        BoundByTanh(low=action_space.low, high=action_space.high),
        DeterministicHead(),
    )

    opt_a = torch.optim.Adam(policy.parameters())
    opt_c = torch.optim.Adam(q_func.parameters())

    rbuf = replay_buffers.ReplayBuffer(10**6)

    explorer = explorers.AdditiveGaussian(
        scale=0.5, low=action_space.low, high=action_space.high
    )
    # explorer = explorers.AdditiveOU(
    #     mu=0.0,
    #     theta=0.15,
    #     sigma=0.3,
    #     start_with_mu=False,
    # )

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(
            np.float32
        )

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = DDPG(
        policy,
        q_func,
        opt_a,
        opt_c,
        rbuf,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_method="soft",
        target_update_interval=1,
        update_interval=1,
        soft_update_tau=5e-3,
        n_times_update=1,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
    )

    if len(args.load) > 0 or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not len(args.load) > 0 or not args.load_pretrained
        if len(args.load) > 0:
            agent.load(args.load)
        else:
            agent.load(
                utils.download_model(
                    "DDPG", args.env, model_type=args.pretrained_type
                )[0]
            )

    eval_env = make_env(test=False)
    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            # env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
        import json
        import os

        with open(os.path.join(args.outdir, "demo_scores.json"), "w") as f:
            json.dump(eval_stats, f)
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_env=eval_env,
            # eval_env=env,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            checkpoint_freq=50000,
            train_max_episode_len=timestep_limit,
        )


if __name__ == "__main__":
    main()
