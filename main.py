import os
import os.path as osp
import itertools
import argparse
import yaml
from pprint import pprint
from agent import get_agent_cls


def main(config_path):
    with open(config_path) as f:
        config = yaml.full_load(f)
        if not config['search']['active']:
            pprint("Configuration File:")
            pprint(config)

    # Construct training agent
    agent_cls = get_agent_cls(config['agent'])
    agent = agent_cls(config)
    if config['train']['resume']:
        agent.resume()

    # Evalution only
    if config['train']['final']:
        keys = [ k
                for k in config['search'].keys()
                if type(config['search'][k]) == list ]
        values = [ config['search'][k] for k in keys ]
        combinations = list(itertools.product(*values))

        if not config['search']['active']:
            agent.finalize()
        else:
            for combination in combinations:
                factors = []
                for idx, v in enumerate(combination):
                    k = keys[idx]
                    factor = f"{k}={v}"
                    factors.append(factor)
                    agent.config['valid'][k] = v
                condition = "[Condition]: " + ", ".join(factors)
                print(condition)
                agent.finalize()

    # Train From Scratch
    else:
        agent.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="path to configuration file")
    args = vars(parser.parse_args())
    main(args['config'])
