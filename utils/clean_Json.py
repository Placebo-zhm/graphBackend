import json

INPUT_JSONFILE="../data/jsondata/txtJsonData.json"
OUTPUT_JSONFILE="../data/jsondata/txtJsonData_clean.json"

def cleanJson():
    # 读取JSON文件
    with open(INPUT_JSONFILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历每个条目
    for entry in data:
        if "CR" in entry:
            # 过滤包含"DOI"的引用条目（不区分大小写）
            filtered_cr = [
                citation for citation in entry["CR"]
                if "DOI" in citation.upper()
            ]
            entry["CR"] = filtered_cr

    # 保存清洗后的数据（覆盖原文件或保存到新文件）
    with open(OUTPUT_JSONFILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

