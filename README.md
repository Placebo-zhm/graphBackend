# 使用说明 #
## data ##
data文件夹下包含两个文件夹  

1. jsondata存储处理后的数据
2. txtdata存储原始文档数据


## utils ##
1. get_Json.py实现读取原始txt文件中各个关键字，将其存储在../data/jsondata/txtJsonData.json中。
2. clean_Json.py实现数据清洗，将上述json文件中CR（引用文献）字段中不包含DOI的清除。
3. Myp2neo4j.py实现将清洗后的json文件，保存于neo4j图数据库中，创建作者和论文的图谱，neo4j的连接需要根据自身情况进行修改。
4. time_sequence.py实现将清洗后的json文件，根据其中的PY（年份）字段和DE（关键词）字段，统计关键词趋势。

test
# 123123 #