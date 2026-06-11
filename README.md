# Eval AI

AI 评测工具与方法仓库，基于 [evalscope](https://github.com/modelscope/evalscope) 构建，覆盖**模型评测**与 **Agent 评测**两大维度。

## 评测体系总览

```
Eval AI
├── 模型评测 (Model Eval)
│   ├── API 兼容性测试 — 验证模型对 OpenAI API 协议的兼容程度
│   ├── 性能测试 — 吞吐量、首 token 延迟、多轮对话性能等
│   └── 精度测试 — 函数调用准确率、指令遵循能力等
│
└── Agent 评测 (Agent Eval)
    ├── Skills 评测 — 单项技能（函数调用、工具使用、多轮规划等）
    └── Agent 评测 — 端到端 Agent 任务完成度（Tau2-Bench、swe-bench 等）、agent执行轨迹的收集等
```

## 目录结构

```
eval_ai/
├── dataset/                     # 评测数据集
│   └── general_fc/              # 函数调用数据集
│       ├── example.jsonl        # 基础示例
│       ├── code_test.jsonl      # 代码相关函数调用
│       ├── complex_test.jsonl   # 复杂场景
│       ├── search_test.jsonl    # 搜索类工具调用
│       ├── full_test.jsonl      # 综合测试
│       ├── multi_turn_test.jsonl # 多轮对话
│       └── multi_tool_test.jsonl # 多工具组合
│
├── perf/                        # 性能评测
│   └── eval_multi_turn/         # 多轮对话性能基准
│       ├── run_multi_turn_benchmark.py
│       └── custom_multi_turn_dataset.py
│
├── agent/                       # Agent 评测
│   └── eval_agent/              # Agent 能力评测
│       ├── run_claude_agent_eval.py   # 评测主脚本
│       ├── create_agent_test_data.py  # 测试数据生成
│       └── claude_agent_eval_guide.md  # 评测指南
│
└── README.md
```

## 快速开始

### 环境准备

```bash
# 安装 evalscope
pip install evalscope

# 设置 API Key
export ANTHROPIC_API_KEY="your-api-key"
```

### 模型评测

#### 1. 函数调用精度评测（General FC）

使用自定义数据集评测模型的函数调用能力：

```bash
# 使用内置示例数据
python agent/eval_agent/run_claude_agent_eval.py --mode general_fc

# 指定本地数据集和子集
python agent/eval_agent/run_claude_agent_eval.py --mode general_fc \
    --data-path dataset/general_fc \
    --subset code_test
```

#### 2. 性能评测（多轮对话）

```bash
cd perf/eval_multi_turn
python run_multi_turn_benchmark.py
```

#### 3. 标准化函数调用评测（BFCL）

```bash
# BFCL v3
python agent/eval_agent/run_claude_agent_eval.py --mode bfcl --bfcl-version v3 --subsets simple

# BFCL v4
python agent/eval_agent/run_claude_agent_eval.py --mode bfcl --bfcl-version v4 --subsets simple parallel
```

### Agent 评测

#### Tau2-Bench 多轮对话任务评测

评测 Agent 在复杂多轮对话场景中的任务完成度：

```bash
# 单领域评测
python agent/eval_agent/run_claude_agent_eval.py --mode tau2_bench --subsets airline

# 多领域评测
python agent/eval_agent/run_claude_agent_eval.py --mode tau2_bench --subsets airline retail telecom
```

#### 运行全部评测

```bash
python agent/eval_agent/run_claude_agent_eval.py --mode all
```

### 自定义模型

```bash
# 使用预设模型别名
python agent/eval_agent/run_claude_agent_eval.py --mode general_fc --model sonnet

# 指定自定义模型 ID
python agent/eval_agent/run_claude_agent_eval.py --mode general_fc --model-id "your-model-id"
```

## 评测方法说明

### 模型评测

| 评测维度 | 评测工具/数据集 | 核心指标 |
|---------|--------------|--------|
| API 兼容性 | OpenAI API 协议对齐测试 | 请求成功率、响应格式合规率 |
| 性能 | 多轮对话基准、自定义压测 | TPS、TTFT、TPOT、并发吞吐 |
| 精度 — 函数调用 | General FC / BFCL | 工具选择准确率、参数填充准确率 |
| 精度 — 指令遵循 | 自定义指令数据集 | 指令遵循率、格式合规率 |

### Agent 评测

| 评测维度 | 评测工具/数据集 | 核心指标 |
|---------|--------------|--------|
| Skills — 函数调用 | General FC / BFCL | 单工具 / 多工具 / 并行调用准确率 |
| Skills — 多轮规划 | multi_turn_test | 对话一致性、上下文保持能力 |
| Agent — 任务完成 | Tau2-Bench | Pass rate（airline / retail / telecom） |

### 数据集格式

评测数据采用 JSONL 格式，每行一条测试样本：

```json
{
  "messages": [
    {"role": "system", "content": "系统提示"},
    {"role": "user", "content": "用户请求"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "function_name",
        "description": "函数描述",
        "parameters": {"type": "object", "properties": {...}}
      }
    }
  ],
  "should_call_tool": true
}
```

可通过 `create_agent_test_data.py` 程序化生成自定义测试数据。

## 扩展评测

添加新的评测维度只需：

1. 在 `dataset/` 下添加对应数据集
2. 在 evalscope 中注册自定义评测器或复用内置评测器
3. 在运行脚本中添加新的评测模式

详见 [Agent 评测指南](agent/eval_agent/claude_agent_eval_guide.md)。

## 致谢

- [evalscope](https://github.com/modelscope/evalscope) — 核心评测框架
- [BFCL](https://github.com/ShishirPatil/gorilla) — 标准化函数调用基准
- [Tau2-Bench](https://github.com/sierra-research/tau2-bench) — Agent 多轮对话评测基准
