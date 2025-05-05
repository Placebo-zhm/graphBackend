import json
from collections import defaultdict

def process_keywords(input_file):
    # 读取原始JSON数据
    with open(input_file, 'r') as f:
        papers = json.load(f)

    # 创建关键词索引字典
    keyword_nodes = defaultdict(lambda: {
        'keyword_name': '',
        'paper_count': 0,
        'related_dois': []
    })

    # 遍历所有论文
    for paper in papers:
        doi = paper.get('DI', '')

        if doi=="empty":
            continue

        # 处理每个关键词
        for keyword in paper.get('keywords', []):
            node = keyword_nodes[keyword]
            node['keyword_name'] = keyword
            node['paper_count'] += 1
            if doi not in node['related_dois']:
                node['related_dois'].append(doi)

    # 转换为标准字典并排序
    sorted_keywords = sorted(
        [dict(node) for node in keyword_nodes.values()],
        key=lambda x: x['paper_count'],
        reverse=True
    )

    return sorted_keywords


if __name__ == '__main__':
    # 输入参数
    input_json = "../data/jsondata/random_kw.json"
    output_json = "../data/jsondata/keyword_nodes.json"

    # 执行处理
    result = process_keywords(input_json)

    # 保存结果
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"成功生成关键词节点，共处理 {len(result)} 个关键词")