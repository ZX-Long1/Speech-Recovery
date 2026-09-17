import argparse
from core import run_resnet

p = argparse.ArgumentParser()
p.add_argument('--split_point', type=int, choices=[1, 3], default=1)
p.add_argument('--smoke', action='store_true')
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--out_dir', type=str, default='')
args = p.parse_args()
run_resnet(dataset='cremad', mode='blackbox', split_point=args.split_point,
           smoke=args.smoke, epochs=args.epochs, gpu=args.gpu,
           out_dir=args.out_dir or None)
