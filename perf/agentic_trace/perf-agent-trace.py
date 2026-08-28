"""
agent-trace 性能测试
requirements:evalscope==1.11.0
"""

from evalscope import run_task
from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
import os

url= os.environ.get('API_URL', '')
apikey = os.environ.get('API_KEY', '')


model_name=os.environ.get('MODEL_NAME', '')

tokenizer_path = '/path_to/tokenizer/'
dataset_path = "/path_to/agent-trace/"

task="agent-trace"
output_dir = f"../outputs/{task}/"

task_perf = Arguments(
    outputs_dir = f"{output_dir}/{model_name}",
    model=model_name,
    url=url,
    api_key=apikey,

    tokenizer_path=tokenizer_path,

    api='openai',
    dataset_path=dataset_path,
    dataset="trie_agentic_coding",
    
    multi_turn=True,
    max_turns = 18,
    # 用户参数覆盖
    parallel=[10],
    number=[10],
    rate = 4,

    stream=True,
    extra_args = {
        "ignore_eos":True
    }

)

results = run_perf_benchmark(task_perf)