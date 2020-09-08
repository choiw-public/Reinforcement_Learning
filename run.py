from functions.handler import Handler
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--game', type=str, default='vegetarian', help="options: vegetarian, bubble")
argparser.add_argument('--phase', type=str, default='train', help="options: manual_play, train, test")
argparser.add_argument('--RL_type', type=str, default='policy_gradient', help="Reinforcement learning type. Options: policy_gradient, 'policy_gradient'")
args = argparser.parse_args()

Handler(args)
