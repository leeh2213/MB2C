import argparse
import yaml

with open('/data/lihao/workspace/MB2C/configs/ThingsEEG.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition training with vit encoder')
parser.add_argument('--dnn', type=str, default=config['training']['dnn'])
parser.add_argument('--epoch', type=int, default=config['training']['epoch'])
parser.add_argument('--num_sub', type=int, default=config['training']['num_sub'])
parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'])
parser.add_argument('--seed', type=int, default=config['training']['seed'])
parser.add_argument('--reproduce', type=bool, default=config['training']['reproduce'])
parser.add_argument('--pretrained', type=bool, default=config['training']['pretrained'])
parser.add_argument('--disp_interval', type=int, default=config['training']['disp_interval'])
parser.add_argument('--n_way', type=int, default=config['training']['n_way'])

parser.add_argument('--lam', type=float, default=config['data_augmentation']['lam'])
parser.add_argument('--MixRatio', type=float, default=config['data_augmentation']['MixRatio'])
parser.add_argument('--is_aug', type=bool, default=config['data_augmentation']['is_aug'])

parser.add_argument('--is_gan', type=bool, default=config['bcwgan']['is_gan'])
parser.add_argument('--cyclelambda', type=float, default=config['bcwgan']['cyclelambda'])
parser.add_argument('--REG_W_LAMBDA', type=float, default=config['bcwgan']['REG_W_LAMBDA'])
parser.add_argument('--REG_Wz_LAMBDA', type=float, default=config['bcwgan']['REG_Wz_LAMBDA'])
parser.add_argument('--GP_LAMBDA', type=float, default=config['bcwgan']['GP_LAMBDA'])
parser.add_argument('--CENT_LAMBDA', type=float, default=config['bcwgan']['CENT_LAMBDA'])
parser.add_argument('--clalambda', type=float, default=config['bcwgan']['clalambda'])
parser.add_argument('--lr', type=float, default=config['bcwgan']['lr'])

args = parser.parse_args()

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
