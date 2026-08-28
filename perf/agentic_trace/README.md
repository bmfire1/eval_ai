# agent-trace 性能测试

基于 evalscope `trie_agentic_coding` 数据集对推理服务进行多轮 agentic workflow 回放压测。通过回放真实生产 agent trace 的 token 长度序列与工具调用时延，测量目标模型在多轮长上下文场景下的吞吐、延迟、首尾 token 等性能指标。

## 测试原理

数据集中的每条 trace 描述一次完整的多轮 agent 对话，由以下字段定义（非真实文本，仅记录 token 长度与等待时间）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `input_prompt_length` | int | 初始 user prompt 的 token 长度 |
| `num_turns` | int | 工具调用轮数（不含最终回复） |
| `assistant_response_length` | List[int] | 每轮 assistant 回复的 `max_tokens` 上限 |
| `tool_call_output_length` | List[int] | 每轮以 user 角色注入的"工具输出" token 长度 |
| `tool_call_latency` | List[float] | 进入下一轮前 sleep 的秒数，模拟工具执行耗时 |
| `final_assistant_response_length` | int | 最终一轮 assistant 回复的 `max_tokens` 上限 |

回放时由 `trie_agentic_coding` 插件合成 token 长度精确匹配的占位文本，共发起 `num_turns + 1` 个请求：

- **Turn 0**：发送初始 prompt，`max_tokens = assistant_response_length[0]`，不 sleep。
- **Turn 1..num_turns-1**：先 sleep `tool_call_latency[i-1]`，再发送长度为 `tool_call_output_length[i-1]` 的工具输出，`max_tokens = assistant_response_length[i]`。
- **Turn num_turns（最终轮）**：sleep 后发送工具输出，`max_tokens = final_assistant_response_length`，标记 `is_final=True`。

为保证长度序列被严格执行，必须开启 `ignore_eos`。

> **数值对齐说明**：上游 `trie` 工具走 `/v1/completions` 发原始 prompt，本测试走 `/v1/chat/completions`，服务端 chat template 每轮会多加约 10-20 tokens，因此 prompt token 数与 cache 命中率相对 `trie` 会有百分之几的偏差，属固有差异。

## 数据集

默认使用 `dataset/agent-trace/agentic_coding_8k.jsonl`（coding-agent，~8k 上下文）。同一数据集目录下还包含 `code_qa_8k.jsonl`、`office_work_8k.jsonl` 两个工作负载，可通过切换 `dataset` 插件复用。原始来源为 [applied-compute/trie](https://github.com/applied-compute/trie)（Apache-2.0），已由 evalscope 重新托管于 ModelScope `evalscope/trie-workloads`。

## 依赖

```text
evalscope==1.11.0
```

> **Tokenizer 匹配要求**：`tokenizer_path` 指定的 tokenizer 必须与被测模型完全一致，或至少为同系列模型。回放时插件按本地的 tokenizer 计数来合成占位文本，若与服务端模型 tokenizer 不一致，服务端实际接收到的 token 数将与 trace 记录的长度序列偏离，导致压测结果失真（各轮 prompt/tool 输出/assistant 回复长度都不准，cache 命中率、吞吐、延迟等指标均不可信）。

数据集插件实现位于 `evalscope/perf/plugin/datasets/trie.py`。

## 运行

### 1. 配置参数

编辑 `perf-agent-trace.py`，按需修改：

- `tokenizer_path`：本地 tokenizer 目录路径，用于精确合成占位文本
- `dataset_path`：指向 `dataset/agent-trace/` 目录
- 压测参数（默认值见下）

### 2. 设置环境变量

```bash
export API_URL=http://<host>:<port>/v1/chat/completions
export API_KEY=<your_api_key>
export MODEL_NAME=<model_id>
```

### 3. 执行

```bash
python perf-agent-trace.py
```

结果默认输出到 `../outputs/agent-trace/<MODEL_NAME>/`。

## 关键参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `api` | `openai` | 走 OpenAI 兼容 `/v1/chat/completions` |
| `dataset` | `trie_agentic_coding` | 数据集插件名 |
| `multi_turn` | `True` | 必须开启，每条 trace 作为一次多轮会话回放 |
| `max_turns` | `18` | 单次会话最大轮数上限 |
| `parallel` | `[10]` | 并发请求级数 |
| `number` | `[10]` | 每个并发下的请求数 |
| `rate` | `4` | 请求注入速率（QPS） |
| `stream` | `True` | 流式返回，便于测量 TTFT |
| `extra_args.ignore_eos` | `True` | 必须开启，确保 `max_tokens` 长度序列被精确执行 |

> 调整 `parallel` / `number` / `rate` 可控制压测强度，`max_turns` 应不小于数据集中最大 `num_turns`，否则会截断 trace。
