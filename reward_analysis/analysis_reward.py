import os
import json
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def count_tool_invocations(prompt, response):
    """
    Count the number of tool invocations in a response.
    
    Args:
        prompt: The prompt text that might contain code tags
        response: The model's response text
        
    Returns:
        int: Number of actual tool invocations
    """
    # 统计response中的工具调用次数
    # 使用<code>和</code>标签之间的内容来判断
    code_blocks = re.findall(r'<code>.*?</code>', response, re.DOTALL)
    
    # 获取<execution_results>的数量来验证工具确实被调用了
    execution_blocks = re.findall(r'<execution_results>.*?</execution_results>', response, re.DOTALL)
    
    # 如果execution_blocks数量少于code_blocks，可能有些工具调用没有执行
    # 以execution_blocks的数量为准，因为这表示真正执行的调用
    if execution_blocks:
        return len(execution_blocks)
    
    return len(code_blocks)


def analyze_reward_logs(log_dir):
    """
    Analyze reward logs to get tool usage statistics and reward distribution.
    
    Args:
        log_dir: Directory containing reward log files
    
    Returns:
        dict: Statistics about the logs
    """
    log_files = glob.glob(os.path.join(log_dir, "*.jsonl"))
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return None
    
    # Initialize statistics
    stats = {
        "total_samples": 0,
        "tool_usage": {},
        "rewards": [],
        "data_sources": {},
        "score_by_tool_usage": {}  # 按工具使用次数统计平均分数
    }
    
    # Process each log file
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    
                    stats["total_samples"] += 1
                    
                    # Record data source
                    data_source = sample.get("data_source", "unknown")
                    stats["data_sources"][data_source] = stats["data_sources"].get(data_source, 0) + 1
                    
                    # Record reward score
                    score = sample.get("score", None)
                    if score is not None:
                        stats["rewards"].append(score)
                    
                    # Count tool invocations
                    if "response" in sample and "prompt" in sample:
                        prompt = sample["prompt"]
                        response = sample["response"]
                        tool_count = count_tool_invocations(prompt, response)
                        
                        # 更新工具使用计数
                        stats["tool_usage"][tool_count] = stats["tool_usage"].get(tool_count, 0) + 1
                        
                        # 按工具使用次数统计分数
                        if score is not None:
                            if tool_count not in stats["score_by_tool_usage"]:
                                stats["score_by_tool_usage"][tool_count] = {"total": 0, "count": 0}
                            stats["score_by_tool_usage"][tool_count]["total"] += score
                            stats["score_by_tool_usage"][tool_count]["count"] += 1
                
                except json.JSONDecodeError:
                    print(f"Error parsing line in {log_file}")
                except Exception as e:
                    print(f"Error processing line: {e}")
    
    # 计算每个工具使用次数的平均分数
    for tool_count, data in stats["score_by_tool_usage"].items():
        if data["count"] > 0:
            data["average"] = data["total"] / data["count"]
    
    return stats


def plot_tool_usage(stats, output_dir):
    """Plot tool usage distribution."""
    counts = sorted(stats["tool_usage"].items())
    if not counts:
        return
    
    x, y = zip(*counts)
    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.xlabel("Number of Tool Invocations")
    plt.ylabel("Count")
    plt.title("Tool Usage Distribution")
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tool_usage_distribution.png"))
    plt.close()


def plot_reward_distribution(stats, output_dir):
    """Plot reward score distribution."""
    if not stats["rewards"]:
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(stats["rewards"], bins=20, alpha=0.7)
    plt.xlabel("Reward Score")
    plt.ylabel("Count")
    plt.title("Reward Score Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "reward_distribution.png"))
    plt.close()


def plot_data_sources(stats, output_dir):
    """Plot data sources distribution."""
    if not stats["data_sources"]:
        return
    
    items = sorted(stats["data_sources"].items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xlabel("Data Source")
    plt.ylabel("Count")
    plt.title("Data Sources Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "data_sources_distribution.png"))
    plt.close()


def plot_score_by_tool_usage(stats, output_dir):
    """Plot average score by tool usage."""
    if not stats["score_by_tool_usage"]:
        return
    
    # 准备数据
    items = sorted([(k, v["average"]) for k, v in stats["score_by_tool_usage"].items() if "average" in v])
    if not items:
        return
    
    x, y = zip(*items)
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.xlabel("Number of Tool Invocations")
    plt.ylabel("Average Score")
    plt.title("Average Score by Tool Usage")
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "score_by_tool_usage.png"))
    plt.close()


def analyze_specific_reward_log(log_file, extreme_tool_usage_threshold=10):
    """
    Analyze a specific reward log file.
    
    Args:
        log_file: Path to the specific log file
        extreme_tool_usage_threshold: 工具调用次数阈值，超过此值的样本被视为极端样本
    
    Returns:
        dict: Statistics about the log and extreme samples
    """
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None
    
    # Initialize statistics
    stats = {
        "total_samples": 0,
        "tool_usage": {},
        "rewards": [],
        "data_sources": {},
        "score_by_tool_usage": {},  # 按工具使用次数统计平均分数
        "extreme_samples": []  # 存储极端样本
    }
    
    # Process the log file
    with open(log_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
                
                stats["total_samples"] += 1
                
                # Record data source
                data_source = sample.get("data_source", "unknown")
                stats["data_sources"][data_source] = stats["data_sources"].get(data_source, 0) + 1
                
                # Record reward score
                score = sample.get("score", None)
                if score is not None:
                    stats["rewards"].append(score)
                
                # Count tool invocations
                if "response" in sample and "prompt" in sample:
                    prompt = sample["prompt"]
                    response = sample["response"]
                    tool_count = count_tool_invocations(prompt, response)
                    
                    # 更新工具使用计数
                    stats["tool_usage"][tool_count] = stats["tool_usage"].get(tool_count, 0) + 1
                    
                    # 按工具使用次数统计分数
                    if score is not None:
                        if tool_count not in stats["score_by_tool_usage"]:
                            stats["score_by_tool_usage"][tool_count] = {"total": 0, "count": 0}
                        stats["score_by_tool_usage"][tool_count]["total"] += score
                        stats["score_by_tool_usage"][tool_count]["count"] += 1
                    
                    # 识别极端样本
                    if tool_count >= extreme_tool_usage_threshold:
                        # 保存极端样本
                        extreme_sample = {
                            "line_num": line_num,
                            "tool_count": tool_count,
                            "score": score,
                            "data_source": data_source,
                            "prompt": prompt,
                            "response": response
                        }
                        if "batch_idx" in sample:
                            extreme_sample["batch_idx"] = sample["batch_idx"]
                        if "ground_truth" in sample:
                            extreme_sample["ground_truth"] = sample["ground_truth"]
                        
                        stats["extreme_samples"].append(extreme_sample)
            
            except json.JSONDecodeError:
                print(f"Error parsing line {line_num} in {log_file}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    # 计算每个工具使用次数的平均分数
    for tool_count, data in stats["score_by_tool_usage"].items():
        if data["count"] > 0:
            data["average"] = data["total"] / data["count"]
    
    return stats


def save_extreme_samples(extreme_samples, output_dir):
    """
    将极端样本保存到文件中，用于进一步分析
    
    Args:
        extreme_samples: 极端样本列表
        output_dir: 输出目录
    """
    if not extreme_samples:
        return
    
    # 按工具调用次数排序
    sorted_samples = sorted(extreme_samples, key=lambda x: x["tool_count"], reverse=True)
    
    # 保存包含所有信息的完整JSON文件
    with open(os.path.join(output_dir, "extreme_samples_full.json"), 'w', encoding='utf-8') as f:
        json.dump(sorted_samples, f, ensure_ascii=False, indent=2)
    
    # 保存精简版本的文本文件，更容易查看
    with open(os.path.join(output_dir, "extreme_samples_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"共发现 {len(sorted_samples)} 个极端样本（工具调用次数过多）\n\n")
        
        for i, sample in enumerate(sorted_samples, 1):
            f.write(f"样本 #{i} (行号: {sample['line_num']})\n")
            f.write(f"工具调用次数: {sample['tool_count']}\n")
            f.write(f"得分: {sample['score']}\n")
            f.write(f"数据源: {sample['data_source']}\n")
            if "batch_idx" in sample:
                f.write(f"批次索引: {sample['batch_idx']}\n")
            
            # 提取问题（通常在prompt末尾）
            try:
                question_match = re.search(r'Question: (.*?)(?:<｜Assistant｜>|$)', sample['prompt'], re.DOTALL)
                if question_match:
                    question = question_match.group(1).strip()
                    f.write(f"问题: {question}\n")
            except:
                pass
            
            # 计算响应长度
            f.write(f"响应长度: {len(sample['response'])} 字符\n")
            
            # 提取工具调用部分
            code_blocks = re.findall(r'<code>(.*?)</code>', sample['response'], re.DOTALL)
            f.write(f"工具调用次数确认: {len(code_blocks)}\n")
            
            if code_blocks:
                f.write("第一次工具调用:\n")
                f.write(f"{code_blocks[0][:200]}...(截断)\n")
                
                f.write("最后一次工具调用:\n")
                f.write(f"{code_blocks[-1][:200]}...(截断)\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"极端样本已保存到：{os.path.join(output_dir, 'extreme_samples_summary.txt')}")
    print(f"完整极端样本数据已保存到：{os.path.join(output_dir, 'extreme_samples_full.json')}")


def analyze_extreme_samples(extreme_samples):
    """
    分析极端样本的特征
    
    Args:
        extreme_samples: 极端样本列表
        
    Returns:
        dict: 极端样本统计信息
    """
    if not extreme_samples:
        return {}
    
    stats = {
        "count": len(extreme_samples),
        "avg_tool_count": sum(s["tool_count"] for s in extreme_samples) / len(extreme_samples),
        "max_tool_count": max(s["tool_count"] for s in extreme_samples),
        "data_sources": {},
        "scores": []
    }
    
    # 统计数据源分布
    for sample in extreme_samples:
        stats["scores"].append(sample["score"])
        source = sample["data_source"]
        stats["data_sources"][source] = stats["data_sources"].get(source, 0) + 1
    
    return stats


def main(log_file, output_dir, extreme_threshold=10):
    """
    Main function to analyze a specific reward log file.
    
    Args:
        log_file: Path to the specific log file
        output_dir: Directory to save output plots
        extreme_threshold: 工具调用次数阈值，超过此值的样本被视为极端样本
    """
    # Analyze logs
    print(f"Analyzing reward log: {log_file}")
    print(f"极端样本阈值: 工具调用 >= {extreme_threshold} 次")
    stats = analyze_specific_reward_log(log_file, extreme_tool_usage_threshold=extreme_threshold)
    
    if not stats:
        print("No data to analyze.")
        return
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    
    print("\nTool Usage:")
    for count, freq in sorted(stats["tool_usage"].items()):
        print(f"  {count} tool invocation(s): {freq} samples ({freq/stats['total_samples']*100:.2f}%)")
    
    print("\nReward Scores:")
    if stats["rewards"]:
        rewards = np.array(stats["rewards"])
        print(f"  Mean: {np.mean(rewards):.4f}")
        print(f"  Median: {np.median(rewards):.4f}")
        print(f"  Min: {np.min(rewards):.4f}")
        print(f"  Max: {np.max(rewards):.4f}")
        print(f"  Std Dev: {np.std(rewards):.4f}")
        
        # 计算0分样本比例
        zero_score_count = sum(1 for score in rewards if score == 0)
        print(f"  Zero score samples: {zero_score_count} ({zero_score_count/len(rewards)*100:.2f}%)")
    
    print("\nData Sources:")
    for source, count in sorted(stats["data_sources"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} samples ({count/stats['total_samples']*100:.2f}%)")
    
    print("\nAverage Score by Tool Usage:")
    for count, data in sorted(stats["score_by_tool_usage"].items()):
        if "average" in data:
            print(f"  {count} tool invocation(s): {data['average']:.4f} (from {data['count']} samples)")
    
    # 处理极端样本
    extreme_samples = stats.get("extreme_samples", [])
    if extreme_samples:
        print(f"\n发现 {len(extreme_samples)} 个极端样本 (工具调用次数 >= {extreme_threshold}):")
        extreme_stats = analyze_extreme_samples(extreme_samples)
        
        print(f"  平均工具调用次数: {extreme_stats['avg_tool_count']:.2f}")
        print(f"  最大工具调用次数: {extreme_stats['max_tool_count']}")
        
        print("  数据源分布:")
        for source, count in sorted(extreme_stats["data_sources"].items(), key=lambda x: x[1], reverse=True):
            percent = count / len(extreme_samples) * 100
            print(f"    {source}: {count} 样本 ({percent:.2f}%)")
        
        # 保存极端样本到文件
        os.makedirs(output_dir, exist_ok=True)
        save_extreme_samples(extreme_samples, output_dir)
    else:
        print(f"\n未发现工具调用次数 >= {extreme_threshold} 的极端样本")
    
    # Generate plots
    print("\nGenerating plots...")
    os.makedirs(output_dir, exist_ok=True)
    plot_tool_usage(stats, output_dir)
    plot_reward_distribution(stats, output_dir)
    plot_data_sources(stats, output_dir)
    plot_score_by_tool_usage(stats, output_dir)
    
    print(f"Analysis complete. Plots saved to {output_dir}")


if __name__ == "__main__":
    # 使用具体的文件路径
    log_file = "/home/ma-user/modelarts/work/weiyu/xueban_v2/AI-Researcher_Dev/AI-Researcher_dev/scripts/logs/reward_logs/reward_samples_20250510_052550_rank0.jsonl"
    output_dir = "/home/ma-user/modelarts/work/weiyu/xueban_v2/AI-Researcher_Dev/AI-Researcher_dev/verl/reward_analysis/plots"
    
    # 设置极端样本阈值
    extreme_threshold = 10
    
    main(log_file, output_dir, extreme_threshold)
