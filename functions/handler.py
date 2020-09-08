class Handler:
    def __init__(self, config):
        if config.game == 'vegetarian':
            if config.RL_type == 'deep_q':
                from functions.deep_q_vegetarian import VegetarianDeepQ
                VegetarianDeepQ(config)
            elif config.RL_type == 'policy_gradient':
                from functions.monte_carlo_policy_gradients_vegetarian import VegetarianPolicyGradient
                VegetarianPolicyGradient(config)
            else:
                raise ValueError('unexpected RL type')
        elif config.game == 'bubble':
            if config.RL_type == 'deep_q':
                from functions.deep_q_bubble import BubbleDeepQ
                BubbleDeepQ(config)
