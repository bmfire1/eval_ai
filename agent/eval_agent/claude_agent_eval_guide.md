# 基于 Claude-CLI 的 Agent 评测指南

本文档介绍如何使用 EvalScope 框架评测 Claude 模型的 Agent 能力，包括函数调用、多轮对话和任务完成等维度。

## 目录

1. [评测方案概览](#1-评测方案概览)
2. [环境准备](#2-环境准备)
3. [方案一：General FC 函数调用评测](#3-方案一general-fc-函数调用评测)
4. [方案二：Tau2-Bench 多轮对话评测](#4-方案二tau2-bench-多轮对话评测)
5. [方案三：BFCL 函数调用评测](#5-方案三bfcl-函数调用评测)
6. [自定义数据集制作](#6-自定义数据集制作)
7. [评测指标说明](#7-评测指标说明)
8. [评测样例](#8-评测样例)
9. [常见问题](#8-常见问题)

---

## 1. 评测方案概览

EvalScope 提供以下 Agent 评测基准：

| 基准 | 评测维度 | 数据来源 | 适用场景 |
|------|----------|----------|----------|
| **General FC** | 函数调用准确性 | 自定义数据集 | 自定义 Agent 工具评测 |
| **Tau2-Bench** | 多轮对话任务完成 | 内置数据集 | 客服类 Agent 评测 |
| **BFCL v3/v4** | 函数调用能力 | 内置数据集 | 通用函数调用评测 |
| **SWE-Bench** | 软件工程任务 | GitHub Issues | 代码 Agent 评测 |
| **ToolBench** | 工具选择能力 | 内置数据集 | 工具选择评测 |

**推荐方案**：
- 快速验证：使用 BFCL 内置数据集
- 自定义场景：使用 General FC + 自定义数据集
- 多轮交互：使用 Tau2-Bench

---

## 2. 环境准备

### 2.1 安装依赖

```bash
cd /{path}/evalscope
pip install -e .
```

### 2.2 配置 API Key

```bash
# Claude API (Anthropic)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# 或使用兼容 OpenAI 格式的服务
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.anthropic.com/v1"
```

### 2.3 Claude 模型 ID

```
claude-opus-4-7      # 最强推理能力
claude-sonnet-4-6    # 平衡性能与速度
claude-haiku-4-5     # 快速响应
```

---

## 3. 方案一：General FC 函数调用评测

适用于自定义 Agent 工具评测，最灵活的方式。

### 3.1 数据集格式

创建 JSONL 文件，每行一个评测样本：

```jsonl
{"messages": [{"role": "user", "content": "帮我查询北京明天的天气"}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "查询指定城市的天气", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名称"}, "date": {"type": "string", "description": "日期"}}}}}], "should_call_tool": true}
```

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `messages` | List[Dict] | 是 | 对话历史，支持多轮 |
| `tools` | List[Dict] | 是 | 工具定义（JSON Schema 格式） |
| `should_call_tool` | bool | 是 | 标签：是否应该调用工具 |

### 3.2 目录结构

```
custom_eval/text/fc/
├── example.jsonl      # 示例数据
└── your_test.jsonl    # 你的测试数据
```

### 3.3 运行评测

```python
from evalscope import run_task
from evalscope.config import TaskConfig
import os

task_cfg = TaskConfig(
    model='claude-sonnet-4-6',
    api_url='https://api.anthropic.com/v1',
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    eval_type='openai_api',
    datasets=['general_fc'],
    dataset_args={
        'general_fc': {
            'local_path': 'custom_eval/text/fc',
            'subset_list': ['example']  # 对应 example.jsonl
        }
    },
    generation_config={
        'temperature': 0,
        'max_tokens': 4096,
    },
    eval_batch_size=1,
)
run_task(task_cfg)
```

---

## 4. 方案二：Tau2-Bench 多轮对话评测

评测 Agent 在多轮对话中完成任务的能力，模拟真实用户交互。

### 4.1 工作原理

```
┌─────────────┐     用户请求      ┌─────────────┐
│  User Model │ ──────────────► │ Agent Model │
│ (模拟用户)   │                 │  (被测模型)  │
└─────────────┘ ◄────────────── └─────────────┘
                  Agent 响应
```

### 4.2 支持的评测领域

- `airline`：航空客服
- `retail`：零售客服
- `telecom`：电信客服

### 4.3 运行评测

```python
from evalscope import run_task
from evalscope.config import TaskConfig
import os

task_cfg = TaskConfig(
    model='claude-sonnet-4-6',  # 被测 Agent
    api_url='https://api.anthropic.com/v1',
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    eval_type='openai_api',
    datasets=['tau2_bench'],
    dataset_args={
        'tau2_bench': {
            'subset_list': ['airline', 'retail', 'telecom'],
            'extra_params': {
                'user_model': 'claude-sonnet-4-6',  # 模拟用户的模型
                'api_key': os.getenv('ANTHROPIC_API_KEY'),
                'api_base': 'https://api.anthropic.com/v1',
                'generation_config': {'temperature': 0.7}
            }
        }
    },
    eval_batch_size=5,
    generation_config={'temperature': 0.6},
)
run_task(task_cfg)
```

### 4.4 评测指标

- `mean_Pass^1`：首次对话完成任务的比例
- 任务完成率
- API 调用正确率
- 业务策略遵循度

---

## 5. 方案三：BFCL 函数调用评测

Berkeley Function Calling Leaderboard，标准化的函数调用评测基准。

### 5.1 支持的子集

| 子集 | 说明 |
|------|------|
| `simple` | 简单函数调用 |
| `live_simple` | 真实场景简单调用 |
| `irrelevance` | 无关查询测试 |
| `multi_turn` | 多轮函数调用 |
| `javascript` | JS 函数调用 |
| `python` | Python 函数调用 |

### 5.2 运行评测

```python
from evalscope import run_task
from evalscope.config import TaskConfig
import os

task_cfg = TaskConfig(
    model='claude-sonnet-4-6',
    api_url='https://api.anthropic.com/v1',
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    eval_type='openai_api',
    datasets=['bfcl_v3'],
    dataset_args={
        'bfcl_v3': {
            'extra_params': {
                'underscore_to_dot': True,  # 函数名格式转换
                'is_fc_model': True,        # 原生函数调用模型
            },
            'subset_list': ['simple', 'multi_turn']
        }
    },
    generation_config={
        'temperature': 0,
        'max_tokens': 64000,
        'parallel_tool_calls': True,  # 并行函数调用
    },
    stream=False,
)
run_task(task_cfg)
```

---

## 6. 自定义数据集制作

### 6.1 定义工具 (Tool Schema)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在知识库中搜索相关信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "执行 Python 代码",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的 Python 代码"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时时间（秒）",
                        "default": 30
                    }
                },
                "required": ["code"]
            }
        }
    }
]
```

### 6.2 创建评测样本

```python
import json

def create_sample(user_query, tools, should_call_tool, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})

    sample = {
        "messages": messages,
        "tools": tools,
        "should_call_tool": should_call_tool
    }
    return json.dumps(sample, ensure_ascii=False)

# 示例
tools = [...]  # 工具定义

samples = [
    # 应该调用工具
    create_sample("搜索关于 Python 异步编程的文档", tools, True),
    # 不应该调用工具（无关查询）
    create_sample("今天天气怎么样？", tools, False),
]

# 保存到文件
with open('custom_eval/text/fc/my_test.jsonl', 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(sample + '\n')
```

### 6.3 多轮对话样本

```python
sample = {
    "messages": [
        {"role": "user", "content": "帮我搜索 Python 教程"},
        {"role": "assistant", "content": "我已经为您搜索到了相关教程..."},
        {"role": "user", "content": "执行一下第一段示例代码"}
    ],
    "tools": tools,
    "should_call_tool": True
}
```

---

## 7. 评测指标说明

### 7.1 General FC 指标

| 指标 | 说明 |
|------|------|
| `count_finish_reason_tool_call` | 正确返回工具调用的样本数 |
| `count_successful_tool_call` | 工具名称和参数都验证通过的样本数 |
| `schema_accuracy` | 参数通过 JSON Schema 验证的比例 |
| `tool_call_f1` | 是否应调用工具的 F1 分数 |

### 7.2 BFCL 指标

| 指标 | 说明 |
|------|------|
| `accuracy` | 整体准确率 |
| `valid_syntax` | 语法正确率 |
| `exact_match` | 完全匹配率 |

### 7.3 Tau2-Bench 指标

| 指标 | 说明 |
|------|------|
| `mean_Pass^1` | 首次完成任务比例 |
| `task_completion_rate` | 任务完成率 |
| `policy_violation_rate` | 策略违规率 |

---
## 8. 评测样例

### general_fc样例

```bash
python eval_claude_agent/run_claude_agent_eval.py --mode general_fc --subset full_test  --data-path ./text/fc/
2026-06-11 15:19:01 - evalscope - WARNING: Deprecated: The `stream` parameter is deprecated and will be removed in v2.0.0. Use `generation_config.stream` instead.

============================================================
开始 General FC 评测
模型: Kimi-K2.6
数据集: {'general_fc': {'local_path': './text/fc/', 'subset_list': ['full_test']}}
============================================================

2026-06-11 15:19:01 - evalscope - INFO: Args: Task config is provided with TaskConfig type.
2026-06-11 15:19:01 - evalscope - INFO: Running with native backend
。。。
。。。
。。。
2026-06-11 15:24:53 - evalscope - INFO: Overall report table: 
┌───────────┬────────────┬───────────────────────────────┬───────────┬───────┬─────────┬─────────┐
│ Model     │ Dataset    │ Metric                        │ Subset    │   Num │   Score │ Cat.0   │
├───────────┼────────────┼───────────────────────────────┼───────────┼───────┼─────────┼─────────┤
│ Kimi-K2.6 │ general_fc │ tool_call_f1                  │ full_test │    31 │  0.9231 │ default │
├───────────┼────────────┼───────────────────────────────┼───────────┼───────┼─────────┼─────────┤
│ Kimi-K2.6 │ general_fc │ count_finish_reason_tool_call │ full_test │    31 │ 20      │ default │
├───────────┼────────────┼───────────────────────────────┼───────────┼───────┼─────────┼─────────┤
│ Kimi-K2.6 │ general_fc │ count_successful_tool_call    │ full_test │    31 │ 20      │ default │
├───────────┼────────────┼───────────────────────────────┼───────────┼───────┼─────────┼─────────┤
│ Kimi-K2.6 │ general_fc │ schema_accuracy               │ full_test │    31 │  1      │ default │
└───────────┴────────────┴───────────────────────────────┴───────────┴───────┴─────────┴─────────┘ 

2026-06-11 15:24:53 - evalscope - INFO: Overall perf table: 
Model      Dataset       Num    Avg Lat  Avg TTFT    Avg TPOT      Avg Thpt    Avg In    Avg Out
                                    (s)  (ms)        (ms)           (tok/s)       Tok        Tok
---------  ----------  -----  ---------  ----------  ----------  ----------  --------  ---------
Kimi-K2.6  general_fc     31    11.3196  -           -                27.23   144.968    308.194 

2026-06-11 15:24:53 - evalscope - INFO: HTML report generated: /{path}/evalscope/custom_eval/outputs/20260611_151901/reports/report.html
```


---

## 9. 常见问题

### Q1: Claude API 格式兼容问题

Claude 原生 API 格式与 OpenAI 略有不同，建议使用 Anthropic 官方 SDK 或 OpenAI 兼容接口。

```python
# 使用 Anthropic SDK
import anthropic
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
```

### Q2: 函数名格式问题

某些评测要求函数名格式转换：
- 设置 `underscore_to_dot: True` 将下划线转为点号
- 设置 `is_fc_model: True` 启用原生函数调用

### Q3: 多工具并行调用

```python
generation_config={
    'parallel_tool_calls': True,  # 支持 parallel tool calls
}
```

### Q4: 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 限制评测数量快速验证
task_cfg = TaskConfig(
    ...,
    limit=10,  # 只评测前 10 条
)
```

---

## 附录：快速开始脚本

参考 `run_claude_agent_eval.py` 获取完整的评测脚本示例。
