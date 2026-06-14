'''
常规基准（GSM8K、AIME、IFEval）变成多轮工具调用任务，评测模型自身的工具使用与多步推理能力。

最小可运行示例：让 DeepSeek-V4-Flash 在求解 GSM8K 时通过 python_exec 验证计算。

trace查看：
 evalscope service --outputs ./outputs/ --host 0.0.0.0 --port 10020
 浏览器打开127.0.0.0:10020 可以查看本地推理每个case的trace信息

'''

from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

import os
CLAUDE_API_URL = 'https://api.siliconflow.cn/' 
def get_api_key():
    """获取 API Key"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("请设置环境变量 ANTHROPIC_API_KEY")
    return api_key

task_config = TaskConfig(
    model='deepseek-ai/DeepSeek-V4-Flash',
    api_url=CLAUDE_API_URL,
    api_key= get_api_key(),
    eval_type='openai_api',
    # datasets=['gsm8k'],
    datasets=['general_fc'],
    datasets_path='./custom_eval/text/fc/complex_test.jsonl',
    limit=5,
    generation_config={'parallel_tool_calls': True},
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        tools=['python_exec'],
        environment='docker',
        environment_extra={'image': 'python:3.11-slim'},
        max_steps=5,
        kwargs={'system_prompt': '使用 python_exec 来验证整个计算过程.'},
    ),
)
run_task(task_config)
