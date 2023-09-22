import argparse
import numpy as np
import hashlib

parser = argparse.ArgumentParser(description='Exp', conflict_handler='resolve')

parser.add_argument('--dataset', type=str, default='PACS')
parser.add_argument('--test_env', type=int, default=0)
parser.add_argument('--method', type=str, default='FedAVG')
parser.add_argument('--total_iters', type=int, default=5000)
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--back_bone', type=str, default='smallcnn',
                    choices=['smallcnn', 'mediumcnn', 'alexnet', 'resnet18', 'resnet50'])
parser.add_argument('--train_split', type=float, default=0.9)
parser.add_argument('--z_dim', type=int, default=512)
parser.add_argument('--num_samples', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--rounds_per_eval', type=int, default=1)
parser.add_argument('--save_checkpoint', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--load_unfinished', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--saved_folder', type=str, default='None')


# hyper paramters
# can use the HparamsGen to auto-generate hyper parameters when performning random search.
# However, if a hparam is specified when running the program, it will always use that value
class HparamsGen(object):
    def __init__(self, name, default, gen_fn=None):
        self.name = name
        self.default = default
        self.gen_fn = gen_fn

    def __call__(self, hparams_gen_seed=0):
        if hparams_gen_seed == 0 or self.gen_fn is None:
            return self.default
        else:
            s = f"{hparams_gen_seed}_{self.name}"
            seed = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2 ** 31)
            return self.gen_fn(np.random.RandomState(seed))


parser.add_argument('--hparams_gen_seed', type=int,
                    default=0)  # if not 0, used as the seed to generate hyper parameters when appicable
parser.add_argument('--E', type=int, default=5, help='number of local update each round')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--L2R_coeff', type=float, default=HparamsGen('L2R_coeff', 1e-2, lambda r: 10 ** r.uniform(-5, -3)))
parser.add_argument('--CMI_coeff', type=float, default=HparamsGen('CMI_coeff', 5e-4, lambda r: 10 ** r.uniform(-5, -3)))
parser.add_argument('--D_beta', type=float, default=1)

# must have for distributed code
parser.add_argument('--dataset_folder', type=str, default='./data/')
parser.add_argument('--experiment_path', type=str, default='./experiment_folder/')
parser.add_argument('--distributed', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--world_size', type=int, default=1)
# for running by torch.distributed
parser.add_argument('--rank', type=int, default=0)
# for slurm
parser.add_argument('--local_rank', type=int, default=0)

# exclude unimportant args when saving the args
unimportant_args = ['save_checkpoint', 'load_unfinished', 'saved_folder', 'dataset_folder', 'experiment_path',
                    'distributed', 'world_size', 'rank', 'local_rank', 'unimportant_args']
