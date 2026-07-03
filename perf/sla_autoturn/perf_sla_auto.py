'''
大模型性能测试-SLA 自动调优
用于测试服务管理平台的llm
使用random数据集
生成prompt的token数量在prefix_length + min-prompt-length和prefix_length + max-prompt-length之间均匀分布，在一次测试中所有请求prefix部分相同。

SLA (Service Level Agreement) 自动调优功能允许用户定义服务质量指标（如延迟、吞吐量），工具将自动调整请求压力（并发数或请求速率），寻找服务能够满足这些指标的最大压力值。

##SLA相关参数
# --sla-auto-tune      是否启用 SLA 自动调优模式
# --sla-variable rate  自动调优的变量 可选：parallel（并发数）、rate（请求速率）
# --sla-params '[{"p99_ttft": "<0.05"}, {"p99_ttft": "<0.01"}]'  SLA 约束条件，JSON 字符串，支持多组约束（AND/OR 逻辑）
# --sla-num-runs 1         每个测试点的重复运行次数（取平均值，减少波动）
# --sla-fixed-parallel 40  在 --sla-variable=rate 时使用的固定并发数；未设置时默认回退到 --sla-upper-bound 以兼容旧行为
# --sla-lower-bound 10     被调优变量的搜索下界
# --sla-upper-bound 40     被调优变量的搜索上界

@2026.06
'''

from typing import Iterator, List, Dict
from evalscope import TaskConfig, run_task


    
model_name='qwen3p6-27B-w8a8-v21' 

########################################################################
####压测任务
########################################################################

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

model_path='{path_to_model_path}/Qwen3p5-4B/'
output_dir = "./outputs"
dataset_path = 'path-to-dataset'

test_url = "api_url"
apikey = "EMPTY"


task_perf = Arguments(
    outputs_dir=f"{output_dir}/{model_name}/",
    model=model_name,     # 模型名称
    tokenizer_path=model_path,  # 模型路径for dummy input
    url=test_url,     # 本地API地址
    api_key=apikey,
    
    log_every_n_query=5, 
    connect_timeout=6000, 
    read_timeout=6000, 
    
    api='openai',
    dataset='custom', # 数据集名称
    dataset_path=dataset_path,
    
    min_tokens=28, # 最小生成token数
    max_tokens=[128,256], # 最大生成token数,可以是一个区间
    # min_prompt_length=3072,
    # max_prompt_length=3072,    

    parallel=4, 
    number=100,
    rate = 4,

    stream=True, # 是否流式输出,
    extra_args={'ignore_eos': True},
    
    ## sla参数
    sla_params=[{"avg_ttft": "<5000"}, {"p99_ttft": "<100"}],
    sla_auto_tune=True,
    sla_variable='parallel',
    sla_upper_bound=60,
    sla_lower_bound=2,
    sla_num_runs=3,
)

results = run_perf_benchmark(task_perf)
