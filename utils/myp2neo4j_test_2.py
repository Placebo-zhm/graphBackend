from neo4j import GraphDatabase
import json

# Neo4j连接配置
URI = "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = "12345678"

class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_data(self, data):
        with self.driver.session() as session:
            for paper in data:
                # 创建论文节点
                session.execute_write(self._create_paper, paper)
                # 处理作者关系
                session.execute_write(self._link_authors, paper)
                # 处理关键词关系
                session.execute_write(self._link_keywords, paper)
                # 处理引用关系
                session.execute_write(self._link_citations, paper)

    @staticmethod
    def _create_paper(tx, paper):
        tx.run("MERGE (p:Paper {doi: $doi})", doi=paper["doi"])

    @staticmethod
    def _link_authors(tx, paper):
        for author in paper["authors"]:
            # 创建作者节点并建立关系
            tx.run("""
                MERGE (a:Author {name: $name})
                WITH a
                MATCH (p:Paper {doi: $doi})
                MERGE (p)-[:AU]->(a)
            """, name=author, doi=paper["doi"])

    @staticmethod
    def _link_keywords(tx, paper):
        for keyword in paper["keywords"]:
            # 创建关键词节点并建立关系
            tx.run("""
                MERGE (k:Keyword {name: $name})
                WITH k
                MATCH (p:Paper {doi: $doi})
                MERGE (p)-[:KW]->(k)
            """, name=keyword.lower(), doi=paper["doi"])

    @staticmethod
    def _link_citations(tx, paper):
        for cited_doi in paper["cited_dois"]:
            # 创建被引论文节点并建立引用关系
            tx.run("""
                MERGE (cited:Paper {doi: $cited_doi})
                WITH cited
                MATCH (citing:Paper {doi: $citing_doi})
                MERGE (citing)-[:CITED]->(cited)
            """, cited_doi=cited_doi, citing_doi=paper["doi"])

if __name__ == "__main__":
    # 读取JSON数据
    with open("../data/jsondata/citation_relations.json", "r") as f:
        data = json.load(f)

    # 连接并导入数据
    importer = Neo4jImporter(URI, USER, PASSWORD)
    try:
        importer.import_data(data)
        print("数据导入成功！")
    except Exception as e:
        print(f"导入失败: {str(e)}")
    finally:
        importer.close()