import json
import matplotlib.pyplot as plt
from collections import defaultdict

#加载JSON数据
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

#关键词规范化处理
def normalize_keywords(keywords):
    return [kw.strip().lower() for kw in keywords if kw.strip()]

#处理数据并统计关键词频率
def process_data(data):
    # 创建嵌套字典：{年份: {关键词: 计数}}
    keyword_stats = defaultdict(lambda: defaultdict(int))

    for entry in data:
        year = entry.get("PY", "Unknown")
        raw_keywords = entry.get("DE", [])

        clean_keywords = normalize_keywords(raw_keywords)

        # 统计所有关键词的年度频率
        for kw in clean_keywords:
            if kw=='empty':
                continue
            else:
                keyword_stats[year][kw] += 1

    return keyword_stats

#保存年度关键词统计到JSON文件
def save_yearly_keywords(keyword_stats, filename="../data/jsondata/time_sequence.json"):

    yearly_data = {
        year: dict(sorted(kws.items(), key=lambda x: (-x[1], x[0])))
        for year, kws in keyword_stats.items()
    }
    # 添加元数据说明
    output = {
        "_metadata": {
            "description": "Yearly keyword frequency statistics",
            "generated_by": "Research Keyword Analyzer v1.0",
            "total_years": len(yearly_data)
        },
        "data": yearly_data
    }

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✅ 数据已保存至 {filename}")
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")

#获取总频次前N的关键词
def get_top_keywords(data, top_n=10):
    # 合并所有年份的统计
    total_counts = defaultdict(int)

    for entry in data:
        keywords = [kw.strip().lower() for kw in entry.get("DE", []) if kw.strip()]
        for kw in keywords:
            if kw=='empty':
                continue
            total_counts[kw] += 1

    # 排序并取前N
    sorted_keywords = sorted(
        total_counts.items(),
        key=lambda x: (-x[1], x[0])  # 先按频次降序，再按字母升序
    )
    return sorted_keywords[:top_n]

#绘制图像
def plot_trend(keyword_stats, top_keywords):
    plt.figure(figsize=(12, 7))

    # 提取关键词名称（假设top_keywords是元组列表）
    keywords = [kw[0] for kw in top_keywords] if top_keywords and isinstance(top_keywords[0], tuple) else top_keywords

    for keyword in keywords:
        trend_data = {}
        for year, kws in keyword_stats.items():
            trend_data[year] = kws.get(keyword, 0)

        sorted_years = sorted(trend_data.keys())
        counts = [trend_data[year] for year in sorted_years]

        plt.plot(sorted_years, counts, marker='o', linestyle='-', label=keyword)

    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Trends of Top Keywords Over Time', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)  # 年份旋转显示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 使用示例
    data = load_data("../data/jsondata/txtJsonData_clean.json")

    target_keyword = "visualization"

    #计算data获取关键词每年的频次
    trend_data = process_data(data)

    #保存数据到json
    save_yearly_keywords(trend_data)

    #获取前十高的关键词
    top_keywords=get_top_keywords(data)

    #绘图
    plot_trend(trend_data, top_keywords)


