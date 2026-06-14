'''
本地用 Claude Code 跑 GSM8K

最小可运行示例：让 DeepSeek-V4-Flash 在求解 GSM8K.

trace查看：
 evalscope service --outputs ./outputs/ --host 0.0.0.0 --port 10020
 浏览器打开127.0.0.0:10020 可以查看本地推理每个case的trace信息

'''

from evalscope import TaskConfig, run_task
from evalscope.agent.external import ExternalAgentConfig

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
    ## gsm8k数据集用于快速启动
    # datasets=['gsm8k'],   
    ## 配置自定义工具调用数据集
    # datasets=['general_fc'],
    # datasets_path='./custom_eval/text/fc/complex_test.jsonl',
    ## 配置swe-bench
    datasets=['swe_bench_verified'],
    
    limit=5,
    generation_config={'parallel_tool_calls': True},

    agent_config=ExternalAgentConfig(
        framework='claude-code',
        environment='local',
        timeout=1800,
        kwargs={
            'auto_install':True
        }

    ),
)
run_task(task_config)
