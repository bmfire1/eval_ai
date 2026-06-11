"""
Claude Agent 评测脚本

支持三种评测模式：
1. General FC - 自定义函数调用评测
2. Tau2-Bench - 多轮对话任务评测
3. BFCL - 标准化函数调用评测

使用方法：
    python custom_eval/run_claude_agent_eval.py --mode general_fc
    python custom_eval/run_claude_agent_eval.py --mode tau2_bench
    python custom_eval/run_claude_agent_eval.py --mode bfcl

环境变量：
    ANTHROPIC_API_KEY - Claude API Key
"""
import argparse
import os
from evalscope import run_task
from evalscope.config import TaskConfig


# Claude 模型配置
# CLAUDE_MODELS = {
#     'opus': 'claude-opus-4-7',
#     'sonnet': 'claude-sonnet-4-6',
#     'haiku': 'claude-haiku-4-5',
# }

CLAUDE_MODELS = {
    'opus': 'Pro/zai-org/GLM-5.1',
    'sonnet': 'Pro/moonshotai/Kimi-K2.6',
    'haiku': 'deepseek-ai/DeepSeek-V4-Flash',
}

# API 配置
CLAUDE_API_URL = 'https://api.siliconflow.cn/'   #'https://api.anthropic.com/v1'


def get_api_key():
    """获取 API Key"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("请设置环境变量 ANTHROPIC_API_KEY")
    return api_key


def run_general_fc_eval(api_key: str, model: str, data_path: str = None, subset: str = None):
    """
    运行 General FC 函数调用评测

    Args:
        api_key: API Key
        model: 模型名称
        data_path: 本地数据集路径
        subset: 子集名称（对应 jsonl 文件名）
    """
    dataset_args = {}

    if data_path and subset:
        # 使用自定义数据集
        dataset_args['general_fc'] = {
            'local_path': data_path,
            'subset_list': [subset]
        }
    else:
        # 使用内置数据集
        dataset_args['general_fc'] = {
            'subset_list': ['test']
        }

    task_cfg = TaskConfig(
        model=model,
        api_url=CLAUDE_API_URL,
        api_key=api_key,
        eval_type='openai_api',
        datasets=['general_fc'],
        dataset_args=dataset_args,
        generation_config={
            'temperature': 0,
            'max_tokens': 4096,
        },
        eval_batch_size=1,
        stream=False,
    )

    print(f"\n{'='*60}")
    print(f"开始 General FC 评测")
    print(f"模型: {model}")
    print(f"数据集: {dataset_args}")
    print(f"{'='*60}\n")

    run_task(task_cfg=task_cfg)


def run_tau2_bench_eval(api_key: str, model: str, subsets: list = None):
    """
    运行 Tau2-Bench 多轮对话评测

    Args:
        api_key: API Key
        model: 模型名称
        subsets: 评测子集列表 ['airline', 'retail', 'telecom']
    """
    if subsets is None:
        subsets = ['airline']  # 默认只跑 airline

    task_cfg = TaskConfig(
        model=model,
        api_url=CLAUDE_API_URL,
        api_key=api_key,
        eval_type='openai_api',
        datasets=['tau2_bench'],
        dataset_args={
            'tau2_bench': {
                'subset_list': subsets,
                'extra_params': {
                    'user_model': model,  # 使用同一模型模拟用户
                    'api_key': api_key,
                    'api_base': CLAUDE_API_URL,
                    'generation_config': {'temperature': 0.7}
                }
            }
        },
        eval_batch_size=5,
        generation_config={'temperature': 0.6},
        stream=False,
    )

    print(f"\n{'='*60}")
    print(f"开始 Tau2-Bench 多轮对话评测")
    print(f"模型: {model}")
    print(f"评测领域: {subsets}")
    print(f"{'='*60}\n")

    run_task(task_cfg=task_cfg)


def run_bfcl_eval(api_key: str, model: str, version: str = 'v3', subsets: list = None):
    """
    运行 BFCL 函数调用评测

    Args:
        api_key: API Key
        model: 模型名称
        version: BFCL 版本 ('v3' 或 'v4')
        subsets: 评测子集列表
    """
    if subsets is None:
        subsets = ['simple']

    dataset_name = f'bfcl_{version}'

    task_cfg = TaskConfig(
        model=model,
        api_url=CLAUDE_API_URL,
        api_key=api_key,
        eval_type='openai_api',
        datasets=[dataset_name],
        dataset_args={
            dataset_name: {
                'extra_params': {
                    'underscore_to_dot': True,  # 函数名格式转换
                    'is_fc_model': True,        # Claude 支持原生函数调用
                },
                'subset_list': subsets
            }
        },
        generation_config={
            'temperature': 0,
            'max_tokens': 64000,
            'parallel_tool_calls': True,  # 支持并行函数调用
        },
        stream=False,
    )

    print(f"\n{'='*60}")
    print(f"开始 BFCL {version} 函数调用评测")
    print(f"模型: {model}")
    print(f"评测子集: {subsets}")
    print(f"{'='*60}\n")

    run_task(task_cfg=task_cfg)


def run_all_evals(api_key: str, model: str):
    """运行所有评测"""
    print(f"\n{'#'*60}")
    print(f"运行全部评测 - 模型: {model}")
    print(f"{'#'*60}\n")

    # 1. General FC 评测
    print("\n[1/3] General FC 评测")
    run_general_fc_eval(
        api_key=api_key,
        model=model,
        data_path='custom_eval/text/fc',
        subset='example'
    )

    # 2. BFCL 评测
    print("\n[2/3] BFCL 评测")
    run_bfcl_eval(
        api_key=api_key,
        model=model,
        version='v3',
        subsets=['simple']
    )

    # 3. Tau2-Bench 评测
    print("\n[3/3] Tau2-Bench 评测")
    run_tau2_bench_eval(
        api_key=api_key,
        model=model,
        subsets=['airline']
    )

    print("\n" + "="*60)
    print("所有评测完成！")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Claude Agent 评测脚本')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['general_fc', 'tau2_bench', 'bfcl', 'all'],
        default='general_fc',
        help='评测模式'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=list(CLAUDE_MODELS.keys()),
        default='sonnet',
        help='Claude 模型变体 (opus/sonnet/haiku)'
    )
    parser.add_argument(
        '--model-id',
        type=str,
        default=None,
        help='自定义模型 ID（覆盖 --model 参数）'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='custom_eval/text/fc',
        help='本地数据集路径 (general_fc 模式)'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='example',
        help='数据集子集名称 (general_fc 模式)'
    )
    parser.add_argument(
        '--subsets',
        type=str,
        nargs='+',
        default=None,
        help='多个子集 (tau2_bench/bfcl 模式)'
    )
    parser.add_argument(
        '--bfcl-version',
        type=str,
        choices=['v3', 'v4'],
        default='v3',
        help='BFCL 版本'
    )

    args = parser.parse_args()

    # 获取 API Key
    api_key = get_api_key()

    # 确定模型 ID
    model_id = args.model_id or CLAUDE_MODELS[args.model]

    # 根据模式运行评测
    if args.mode == 'general_fc':
        run_general_fc_eval(
            api_key=api_key,
            model=model_id,
            data_path=args.data_path,
            subset=args.subset
        )
    elif args.mode == 'tau2_bench':
        run_tau2_bench_eval(
            api_key=api_key,
            model=model_id,
            subsets=args.subsets
        )
    elif args.mode == 'bfcl':
        run_bfcl_eval(
            api_key=api_key,
            model=model_id,
            version=args.bfcl_version,
            subsets=args.subsets
        )
    elif args.mode == 'all':
        run_all_evals(api_key=api_key, model=model_id)


if __name__ == '__main__':
    main()
