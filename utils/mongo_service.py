import pandas as pd
import numpy as np
from pymongo import MongoClient

# 配置参数（需要根据实际情况修改）
PARQUET_PATH = "../data/parquetdata/create_final_text_units.parquet"
MONGODB_URL = "mongodb://localhost:27017/"
DATABASE_NAME = "parquetDB"
COLLECTION_NAME = "final_text_units"


# 读取Parquet文件
def read_parquet_file(file_path: str):
    df = pd.read_parquet(file_path)

    # 处理ndarray类型的列（替换NaN为空）
    list_columns = ['document_ids', 'entity_ids',
                    'relationship_ids', 'covariate_ids']
    for col in list_columns:
        df[col] = df[col].apply(
            lambda x:(
                x.tolist()
                if isinstance(x, np.ndarray)
                else []
            )
        )

    return df


# 写入MongoDB
def write_to_mongodb(dataframe, url, db_name, collection_name ):
    # 创建MongoDB连接
    client = MongoClient(url)
    db = client[db_name]
    collection = db[collection_name]

    # 转换DataFrame为字典列表
    documents = dataframe.to_dict('records')

    # 批量插入数据
    result = collection.insert_many(documents)
    return len(result.inserted_ids)


if __name__ == "__main__":
    # 配置参数（需要根据实际情况修改）

    # 执行流程
    df = read_parquet_file(PARQUET_PATH)

    inserted_count = write_to_mongodb(
        df,
        MONGODB_URL,
        DATABASE_NAME,
        COLLECTION_NAME
    )
    print(f"成功插入 {inserted_count} 条文本块")