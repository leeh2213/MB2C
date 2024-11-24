import argparse
import yaml

with open('/data/lihao/workspace/MB2C/configs/EEGCVPR40.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition training with EEGCVPR40 dataset')
parser.add_argument('--base_path', type=str, default=config['path']['base_path'])
parser.add_argument('--train_path', type=str, default=config['path']['train_path'])
parser.add_argument('--validation_path', type=str, default=config['path']['validation_path'])
parser.add_argument('--test_path', type=str, default=config['path']['test_path'])

parser.add_argument('--embedding_dim', type=int, default=config['training']['embedding_dim'])
parser.add_argument('--projection_dim', type=int, default=config['training']['projection_dim'])
parser.add_argument('--input_channels', type=int, default=config['training']['input_channels'])
parser.add_argument('--num_layers', type=int, default=config['training']['num_layers'])
parser.add_argument('--batch_size', type=bool, default=config['training']['batch_size'])
parser.add_argument('--test_batch_size', type=bool, default=config['training']['test_batch_size'])
parser.add_argument('--epoch', type=int, default=config['training']['epoch'])
parser.add_argument('--temperature', type=int, default=config['training']['temperature'])
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.hidden_size = args.embedding_dim // 2

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
