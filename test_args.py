import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--image',default="data/train/images")

args = parser.parse_args()

print(args)