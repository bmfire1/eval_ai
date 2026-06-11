"""
Agent 评测数据集生成工具

快速创建 function calling 评测数据集
"""
import json
import os
from typing import List, Dict, Any


# ========== 工具定义模板 ==========

# 搜索工具
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "在知识库中搜索相关信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题"
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
}

# 代码执行工具
CODE_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": "执行 Python 代码并返回结果",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码"
                },
                "language": {
                    "type": "string",
                    "description": "编程语言",
                    "enum": ["python", "javascript"],
                    "default": "python"
                }
            },
            "required": ["code"]
        }
    }
}

# 天气查询工具
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                },
                "date": {
                    "type": "string",
                    "description": "日期，格式：YYYY-MM-DD"
                }
            },
            "required": ["city"]
        }
    }
}

# 数据库查询工具
DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "执行 SQL 查询",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL 查询语句"
                },
                "database": {
                    "type": "string",
                    "description": "数据库名称",
                    "enum": ["users", "products", "orders"]
                }
            },
            "required": ["sql", "database"]
        }
    }
}

# 文件操作工具
FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "file_operation",
        "description": "执行文件操作",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "操作类型",
                    "enum": ["read", "write", "delete", "list"]
                },
                "path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "写入内容（仅 write 操作需要）"
                }
            },
            "required": ["operation", "path"]
        }
    }
}

# 数学计算工具
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式"
                }
            },
            "required": ["expression"]
        }
    }
}


# ========== 测试样本生成 ==========

def create_sample(
    messages: List[Dict],
    tools: List[Dict],
    should_call_tool: bool,
    system_prompt: str = None
) -> Dict[str, Any]:
    """创建单个评测样本"""
    full_messages = []

    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})

    full_messages.extend(messages)

    return {
        "messages": full_messages,
        "tools": tools,
        "should_call_tool": should_call_tool
    }


def generate_search_samples() -> List[Dict]:
    """生成搜索工具评测样本"""
    tools = [SEARCH_TOOL]
    samples = []

    # 应该调用工具的场景
    should_call_queries = [
        "帮我搜索关于 Python 异步编程的资料",
        "查找一下最近的 AI 新闻",
        "搜索如何学习机器学习",
        "帮我找一下 Docker 的使用教程",
        "查询一下 Kubernetes 的最佳实践",
    ]

    for query in should_call_queries:
        samples.append(create_sample(
            messages=[{"role": "user", "content": query}],
            tools=tools,
            should_call_tool=True
        ))

    # 不应该调用工具的场景
    should_not_call_queries = [
        "今天天气怎么样？",
        "给我讲个笑话",
        "你好，你是谁？",
        "你觉得 AI 会取代人类吗？",
        "推荐几部好看的电影",
    ]

    for query in should_not_call_queries:
        samples.append(create_sample(
            messages=[{"role": "user", "content": query}],
            tools=tools,
            should_call_tool=False
        ))

    return samples


def generate_code_samples() -> List[Dict]:
    """生成代码执行评测样本"""
    tools = [CODE_EXEC_TOOL]
    samples = []

    # 应该执行代码的场景
    should_call_queries = [
        "帮我计算斐波那契数列的第 10 项",
        "写一个快速排序算法并测试",
        "计算 1 到 100 的和",
        "生成一个随机密码",
        "解析这个 JSON 字符串",
    ]

    for query in should_call_queries:
        samples.append(create_sample(
            messages=[{"role": "user", "content": query}],
            tools=tools,
            should_call_tool=True
        ))

    # 不应该执行代码的场景
    should_not_call_queries = [
        "什么是 Python？",
        "解释一下什么是递归",
        "给我讲个编程笑话",
        "你觉得哪种编程语言最好？",
    ]

    for query in should_not_call_queries:
        samples.append(create_sample(
            messages=[{"role": "user", "content": query}],
            tools=tools,
            should_call_tool=False
        ))

    return samples


def generate_multi_tool_samples() -> List[Dict]:
    """生成多工具场景评测样本"""
    tools = [SEARCH_TOOL, CODE_EXEC_TOOL, WEATHER_TOOL, CALCULATOR_TOOL]
    samples = []

    # 应该调用特定工具的场景
    multi_tool_queries = [
        ("搜索 Python 教程", True),
        ("计算 123 * 456", True),
        ("北京明天天气怎么样", True),
        ("执行这段代码：print('hello')", True),
        ("今天心情不错", False),
        ("介绍一下你自己", False),
    ]

    for query, should_call in multi_tool_queries:
        samples.append(create_sample(
            messages=[{"role": "user", "content": query}],
            tools=tools,
            should_call_tool=should_call
        ))

    return samples


def generate_multi_turn_samples() -> List[Dict]:
    """生成多轮对话评测样本"""
    tools = [SEARCH_TOOL, CODE_EXEC_TOOL]
    samples = []

    # 多轮对话：用户追问
    samples.append(create_sample(
        messages=[
            {"role": "user", "content": "帮我搜索 Python 异步编程"},
            {"role": "assistant", "content": "我已经为您搜索到了关于 Python 异步编程的相关资料..."},
            {"role": "user", "content": "执行一下第一个示例代码"}
        ],
        tools=tools,
        should_call_tool=True
    ))

    # 多轮对话：上下文相关但不需要工具
    samples.append(create_sample(
        messages=[
            {"role": "user", "content": "帮我搜索 Python 教程"},
            {"role": "assistant", "content": "搜索完成，找到了 10 个相关教程..."},
            {"role": "user", "content": "谢谢，这些教程看起来很有用"}
        ],
        tools=tools,
        should_call_tool=False
    ))

    # 多轮对话：需要新工具调用
    samples.append(create_sample(
        messages=[
            {"role": "user", "content": "搜索机器学习入门"},
            {"role": "assistant", "content": "已找到相关资料..."},
            {"role": "user", "content": "计算一下 2 的 10 次方"}
        ],
        tools=tools,
        should_call_tool=True
    ))

    return samples


def generate_complex_samples() -> List[Dict]:
    """生成复杂场景评测样本"""
    tools = [DATABASE_TOOL, FILE_TOOL, CODE_EXEC_TOOL]
    samples = []

    # 数据库查询场景
    samples.append(create_sample(
        messages=[{"role": "user", "content": "查询 users 表中所有年龄大于 18 的用户"}],
        tools=tools,
        should_call_tool=True
    ))

    # 文件操作场景
    samples.append(create_sample(
        messages=[{"role": "user", "content": "读取 /etc/config.json 文件内容"}],
        tools=tools,
        should_call_tool=True
    ))

    # 多步骤任务
    samples.append(create_sample(
        messages=[
            {"role": "user", "content": "帮我完成以下任务：读取 data.txt 文件，然后统计其中的单词数量"}
        ],
        tools=tools,
        should_call_tool=True,
        system_prompt="你是一个能够执行文件操作和代码计算的助手。"
    ))

    return samples


def generate_all_samples() -> List[Dict]:
    """生成所有评测样本"""
    all_samples = []

    all_samples.extend(generate_search_samples())
    all_samples.extend(generate_code_samples())
    all_samples.extend(generate_multi_tool_samples())
    all_samples.extend(generate_multi_turn_samples())
    all_samples.extend(generate_complex_samples())

    return all_samples


def save_dataset(samples: List[Dict], output_path: str):
    """保存数据集到 JSONL 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"已保存 {len(samples)} 条样本到: {output_path}")


def main():
    # 输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'text', 'fc')
    os.makedirs(output_dir, exist_ok=True)

    # 生成各类样本
    print("生成评测数据集...")

    # 1. 搜索工具样本
    search_samples = generate_search_samples()
    save_dataset(search_samples, os.path.join(output_dir, 'search_test.jsonl'))

    # 2. 代码执行样本
    code_samples = generate_code_samples()
    save_dataset(code_samples, os.path.join(output_dir, 'code_test.jsonl'))

    # 3. 多工具样本
    multi_tool_samples = generate_multi_tool_samples()
    save_dataset(multi_tool_samples, os.path.join(output_dir, 'multi_tool_test.jsonl'))

    # 4. 多轮对话样本
    multi_turn_samples = generate_multi_turn_samples()
    save_dataset(multi_turn_samples, os.path.join(output_dir, 'multi_turn_test.jsonl'))

    # 5. 复杂场景样本
    complex_samples = generate_complex_samples()
    save_dataset(complex_samples, os.path.join(output_dir, 'complex_test.jsonl'))

    # 6. 完整数据集
    all_samples = generate_all_samples()
    save_dataset(all_samples, os.path.join(output_dir, 'full_test.jsonl'))

    print("\n数据集生成完成！")
    print(f"输出目录: {output_dir}")
    print("\n可以使用以下命令运行评测：")
    print(f"  python custom_eval/run_claude_agent_eval.py --mode general_fc --data-path custom_eval/text/fc --subset search_test")


if __name__ == '__main__':
    main()
