import argparse as arg

parser = arg.ArgumentParser()

parser.add_argument("--num-nodes", default=1, type=int)

args = parser.parse_args()

print(args.num_nodes)