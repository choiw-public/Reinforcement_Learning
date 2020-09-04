from functions.handler import Handler
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--game', type=str, default='bubble', help="options: vegetarian, bubble")
argparser.add_argument('--phase', type=str, default='manual_play', help="options: manual_play, train, test")
argparser.add_argument('--RL_type', type=str, default='deep_q', help="Reinforcement learning type. Options: 'deep_q' for now")
args = argparser.parse_args()

Handler(args)
