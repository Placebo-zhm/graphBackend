import json
import re
from clean_Json import cleanJson
from pathlib import Path

#读取txt文件生成json数据

# 使用正则表达式提取字段
FIELD_PATTERN = re.compile(r'^([A-Z][A-Z0-9])\s+(.*)$')  # 匹配两个大写字母开头或者一个大写一个数字的字段
MULTI_LINE_FIELDS = {'AU','AF','TI','CT','ID','C1','C3','EM','OI','FU','FX','CR'}  # 定义多行字段，多行字段为带有多行文本的字段值，要一直读取到下一个关键字
LIST_FIELDS = {'AU','AF','DE','CR'}   #列表存储如作者，引用文献

def extract_info_from_block(block):
    """使用正则表达式提取字段，固定顺序"""
    # info = {field: [] for field in ["AF", "TI", "C3", "CR", "PY", "DI"]}
    field_list=["PT","AU","AF","TI","SO","LA","DT",
                "CT","CY","CL","DE","ID","AB","C1",
                "C3","RP","EM","RI","OI","FU","FX",
                "CR","NR","TC","Z9","U1","U2","PU",
                "PI","PA","SN","EI","J9","JI","PD",
                "PY","VL","IS","BP","EP","DI","PG",
                "WC","WE", "SC","GA","UT","PM","DA"]
    info = {field: [] for field in field_list}
    current_field = None

    for line in block.split('\n'):
        line = line.strip()

        # 匹配字段起始行
        if match := FIELD_PATTERN.match(line):
            field = match.group(1)
            # 处理预定义的字段
            current_field = field if field in info else None
            if current_field:
                info[current_field].append(match.group(2))
        # 处理多行字段的延续
        elif current_field in MULTI_LINE_FIELDS and line:
            info[current_field].append(line)
        # 处理单行字段
        elif current_field and line:
            info[current_field].append(line)
            current_field = None  # 单行字段只捕获一次

    # 后处理逻辑
    processed = {}
    for k, v in info.items():
        # 多行字段用空格连接，单行字段取第一个值
        if k in LIST_FIELDS:
            # 直接保存列表，过滤空值
            if k == 'DE':
                # 合并后按中文分号分割
                combined = " ".join(v)
                processed[k] = [s.strip() for s in combined.split(";") if s.strip()]
            else:
                processed[k] = [s.strip() for s in v if s.strip()]
        elif k in MULTI_LINE_FIELDS:
            processed[k] = " ".join(v).strip()
        else:
            processed[k] = v[0] if v else ""

    for key in processed:
        # 判断空列表或空字符串
        if processed[key] == [] or processed[key] == "":
            if key in LIST_FIELDS:
                processed[key] = ["empty"]
            else:
                processed[key] = "empty"

    return processed


def read_and_split_file(file_path: str, delimiter: str) -> list:
    """使用生成器逐块读取文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            current_block = []
            for line in f:
                line = line.rstrip('\n')  # 保留行尾空格
                if line == delimiter:
                    if current_block:
                        yield '\n'.join(current_block)
                        current_block = []
                else:
                    current_block.append(line)
            if current_block:
                yield '\n'.join(current_block)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []


#自动发现数据文件
# def find_data_files(data_dir: str = "./data") -> list:
#     """自动发现数据目录下的txt文件"""
#     return [
#         str(p) for p in Path(data_dir).glob("*.txt")
#         if p.is_file() and p.suffix == ".txt"
#     ]


# 主函数
def main(output_file: str = "../data/jsondata/txtJsonData.json"):
    # 自动发现数据文件
    #files = find_data_files()
    # 手动添加数据文件
    files = [
        '../data/txtdata/data1.txt',
        '../data/txtdata/data2.txt',
        '../data/txtdata/data3.txt',
        '../data/txtdata/data4.txt',
        '../data/txtdata/data5.txt',
        '../data/txtdata/data6.txt',
        '../data/txtdata/data7.txt',
        '../data/txtdata/data8.txt',
        '../data/txtdata/data9.txt',
        '../data/txtdata/data10.txt',
        '../data/txtdata/data11.txt'
    ]
    if not files:
        print("No data files found in ./data directory!")
        return

    results = []
    #file标记第几个txt文件，block标记的第几块文本
    for file_id, file_path in enumerate(files, 1):
        try:
            for block_id, block in enumerate(read_and_split_file(file_path, "ER"), 1):
                extracted = extract_info_from_block(block)
                results.append({
                    "file_id": file_id,
                    "block_id": block_id,
                    **extracted
                })
        except Exception as e:
            print(f"Skipped {file_path} due to error: {str(e)}")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {len(results)} records to {output_file}")

    #数据清洗，去除CR中不包含DOI的项
    cleanJson()


if __name__ == "__main__":
    # 可以添加命令行参数处理
    main()