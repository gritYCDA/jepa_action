import argparse
import logging
import os
import pprint
import sys
import yaml
import torch
import torch.distributed as dist
from evals.config_utils import load_config
from evals.scaffold import main as eval_main

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str,
        help='path to config file',
        required=True)
    parser.add_argument(
        '--env', type=str, 
        choices=['local', 'azure'], 
        default='azure',
        help='environment to run in (local or azure)')
    parser.add_argument(
        '--tag', type=str,
        help='override tag in config file',
        default="TEST")
    parser.add_argument(
        '--local_rank', type=int,
        help='local rank for distributed training',
        default=0)
    parser.add_argument(
        '--launcher', type=str,
        default='pytorch',
        help='launcher type')
    parser.add_argument(
        '--port', type=str,
        help='port for distributed training',
        default='29500')
    
    # Add cfg-options
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    return parser.parse_args()


class DictAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary.
    """
    @staticmethod
    def parse_value(value):
        """Convert string value to proper type (bool, int, float, or string)"""
        # Handle boolean values
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
            
        # Handle numeric values
        try:
            # Try converting to int first
            return int(value)
        except ValueError:
            try:
                # Try converting to float if int fails
                return float(value)
            except ValueError:
                # Return original string if both conversions fail
                return value
    
    def __call__(self, parser, namespace, values, option_string=None):
        dict_args = {}
        for value in values:
            key = value.split('=')[0]
            value = '='.join(value.split('=')[1:])
            
            # Convert list/tuple strings to actual lists/tuples
            if value.startswith('[') and value.endswith(']'):
                try:
                    value = eval(value)
                except:
                    # If eval fails, keep original string
                    pass
            elif ',' in value:
                value = [self.parse_value(x.strip()) for x in value.split(',')]
                if len(value) == 1:
                    value = value[0]
            else:
                value = self.parse_value(value)
            
            # Handle nested dict (e.g., optimization.lr=0.001)
            key_parts = key.split('.')
            current_dict = dict_args
            for part in key_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            current_dict[key_parts[-1]] = value
            
        setattr(namespace, self.dest, dict_args)

def main():
    # Initialize wandb environment variables -> prevent ID conflicts
    wandb_vars = [var for var in os.environ if var.startswith('WANDB_')]
    for var in wandb_vars:
        os.environ.pop(var)
    # Set wandb account key
    os.environ['WANDB_API_KEY'] = '3f7b0e5db495d33d26adf24bd4f075c6b1c0cbe3'
    args = parse_args()

    # Check if running with torchrun
    if torch.cuda.device_count() > 1:
        # Distributed mode
        # 다중 GPU: torchrun으로 실행된 경우
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

        # 분산 학습에 맞는 로깅 설정: logging level은 계층적: ERROR < WARNING < INFO
        if dist.get_rank() == 0:
            logger.setLevel(logging.INFO)  # INFO, WARNING, ERROR 모두 표시
        else:
            logger.setLevel(logging.ERROR) # WARNING, ERROR만 표시
    else:
        # Single GPU mode
        # 단일 GPU: 일반 python으로 실행된 경우
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        dist.init_process_group(backend='nccl', world_size=1, rank=0)
        torch.cuda.set_device(0)

    # Load and process config
    config = load_config(args.config, args.env)
    
    # Override tag if provided
    if args.tag is not None:
        config['tag'] = args.tag
        logger.info(f'Overriding tag with: {args.tag}')
    
    # Update config with values from cfg-options
    if args.cfg_options is not None:
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        update_nested_dict(config, args.cfg_options)
        logger.info(f'Config updated with CLI options')
    
    logger.info('Loaded config:')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    
    # Launch evaluation
    eval_main(
        config['eval_name'],
        args_eval=config,
        resume_preempt=False)

if __name__ == '__main__':
    main()

"""
Upgrade --cfg-options

<Single GPU>
python -m evals.main_auto --config configs/evals/vitl16_ssv2.yaml --env local --tag my_tag --cfg-options optimization.lr=0.0003 optimization.batch_size=16 pretrain.freeze_all=true

<Multi GPU>
# GPUS=8 ./run_distributed.sh configs/evals/vitl16_ssv2.yaml azure TEST --cfg-options optimization.lr=0.0003 optimization.batch_size=16 pretrain.freeze_all=true

주요 기능:

1. 단일 값 수정: key=value 형식
2. 리스트 값 수정: key=[a,b,c] 또는 key=a,b,c 형식
3. 중첩된 설정 수정: optimization.lr=0.001 형식
4. 여러 설정 동시 수정: 공백으로 구분하여 여러 key=value 쌍 전달 가능

이렇게 수정하면 config 파일을 직접 수정하지 않고도 CLI에서 원하는 설정값을 변경할 수 있습니다.
"""