#!/bin/bash

# 测试不同搜索API端点的结果
# 作者: AI-Researcher团队
# 日期: $(date +%Y-%m-%d)

# 设置搜索URLs
declare -a SEARCH_URLS=(
    "http://192.168.200.194:30014/retrieve"
    "http://192.168.200.194:30013/retrieve"
)

# 测试查询列表
declare -a TEST_QUERIES=(
    "What is machine learning"
    "who won the world cup in 2022"
    "when was the last year thanksgiving was on the 23rd"
    "where did the atomic bomb drop on hiroshima"
)

# 创建输出目录
OUTPUT_DIR="search_test_results"
mkdir -p $OUTPUT_DIR

# 显示时间戳函数
timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

# 打印分隔线
print_separator() {
    echo "========================================================"
}

# 记录开始时间
START_TIME=$(date +%s)
echo "$(timestamp) - 开始测试搜索API"
print_separator

# 对每个URL进行测试
for SEARCH_URL in "${SEARCH_URLS[@]}"; do
    URL_BASENAME=$(basename $SEARCH_URL)
    echo "$(timestamp) - 测试端点: $SEARCH_URL"
    
    # 检查API是否在线
    echo "检查API是否可用..."
    if ! curl -s --connect-timeout 5 --max-time 10 -X POST "$SEARCH_URL" \
        -H "Content-Type: application/json" \
        -d '{"queries": ["test"], "topk": 1, "return_scores": true}' > /dev/null; then
        echo "$(timestamp) - 警告: API端点 $SEARCH_URL 不可用，跳过此测试"
        print_separator
        continue
    fi
    
    # 对每个测试查询发送请求
    for QUERY in "${TEST_QUERIES[@]}"; do
        QUERY_FILE="${OUTPUT_DIR}/${URL_BASENAME}_$(echo "$QUERY" | tr ' ' '_').json"
        FORMATTED_FILE="${OUTPUT_DIR}/${URL_BASENAME}_$(echo "$QUERY" | tr ' ' '_')_formatted.txt"
        
        echo "$(timestamp) - 查询: \"$QUERY\""
        
        # 发送查询请求并保存完整响应
        echo "发送查询到 $SEARCH_URL..."
        curl -s -X POST "$SEARCH_URL" \
            -H "Content-Type: application/json" \
            -d "{\"queries\": [\"$QUERY\"], \"topk\": 3, \"return_scores\": true}" \
            > "$QUERY_FILE"
        
        # 检查是否成功获取结果
        if [ $? -ne 0 ] || [ ! -s "$QUERY_FILE" ]; then
            echo "$(timestamp) - 错误: 查询\"$QUERY\"失败或返回空结果"
            continue
        fi
        
        # 显示结果摘要
        echo "查询结果已保存到: $QUERY_FILE"
        
        # 提取并格式化返回结果，模拟代码中的处理
        echo "格式化结果..."
        echo "查询: \"$QUERY\"" > "$FORMATTED_FILE"
        echo "" >> "$FORMATTED_FILE"
        echo "格式化搜索结果:" >> "$FORMATTED_FILE"
        echo "" >> "$FORMATTED_FILE"
        
        # 使用jq解析JSON并提取搜索结果
        if command -v jq &> /dev/null; then
            # 检查JSON结构
            if jq -e '.result' "$QUERY_FILE" > /dev/null 2>&1; then
                # 提取并格式化搜索结果
                jq -r '.result[0][] | "Doc \(.document.title // "无标题"):\n\(.document.contents)\n"' "$QUERY_FILE" >> "$FORMATTED_FILE" 2>/dev/null
                
                # 如果jq命令失败，使用备用方法提取信息
                if [ $? -ne 0 ]; then
                    echo "无法通过jq解析搜索结果，使用备用方法..." >> "$FORMATTED_FILE"
                    jq -r '.' "$QUERY_FILE" >> "$FORMATTED_FILE"
                fi
            else
                echo "结果JSON格式不符合预期，原始内容:" >> "$FORMATTED_FILE"
                cat "$QUERY_FILE" >> "$FORMATTED_FILE"
            fi
        else
            echo "jq命令不可用，无法格式化JSON，原始结果:" >> "$FORMATTED_FILE"
            cat "$QUERY_FILE" >> "$FORMATTED_FILE"
        fi
        
        # 显示前5行格式化结果
        echo ""
        echo "格式化结果预览 (保存到 $FORMATTED_FILE):"
        head -n 5 "$FORMATTED_FILE"
        echo "..."
        
        # 休息一下，避免请求过快
        sleep 1
    done
    
    print_separator
done

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "$(timestamp) - 测试完成，总耗时: ${DURATION}秒"
echo "结果文件保存在: $(realpath $OUTPUT_DIR)"

# 创建摘要报告
SUMMARY_FILE="${OUTPUT_DIR}/summary_report.txt"
echo "搜索API测试摘要" > "$SUMMARY_FILE"
echo "测试时间: $(timestamp)" >> "$SUMMARY_FILE"
echo "测试端点: ${SEARCH_URLS[*]}" >> "$SUMMARY_FILE"
echo "测试查询: ${TEST_QUERIES[*]}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "查询结果文件列表:" >> "$SUMMARY_FILE"
ls -l $OUTPUT_DIR/*.json | awk '{print "- " $9}' >> "$SUMMARY_FILE"

echo "测试摘要已保存到: $SUMMARY_FILE"

# 给出分析建议
echo ""
echo "分析步骤建议:"
echo "1. 检查不同端点返回的JSON格式是否一致"
echo "2. 查看格式化后的结果是否符合预期"
echo "3. 比较返回结果与原始查询的相关性"
echo ""
echo "如有需要，可以通过修改脚本中的TEST_QUERIES数组添加更多测试查询" 