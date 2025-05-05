#from p2neo4j import GraphDatabase
from neo4j import GraphDatabase
import json
import re

# Neo4j连接配置
URI = "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = "12345678"

# 初始化驱动
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# 增强的DOI正则表达式（兼容更多格式）
CR_PATTERN = re.compile(
    r"""
    ^
    (?P<author>(.*?)\s*),\s*       # 作者（第一个逗号前的内容）
    (?P<year>\d{4}),\s*         # 年份（4位数字）
    (?P<journal>(.*?)\s*),\s*      # 期刊名称
    (?:V(?P<volume>\d+),\s*)?   # 卷号（可选）
    (?:P(?P<page>\d+),\s*)?     # 页码（可选）
    (?:.*?DOI\s+(?P<doi>10\.\d+/[^\s\],;]+))
    """,
    re.IGNORECASE | re.VERBOSE
)

def extract_doi(cr_entry):
    """从CR条目中提取并标准化DOI"""
    match = CR_PATTERN.search(cr_entry.strip())
    if not match:
        return

    author = match.group("author").strip()
    year = int(match.group("year"))
    journal = match.group("journal").strip()
    volume = match.group("volume")
    page = match.group("page")
    cited_doi = match.group("doi").strip()
    return{
        "author": author,
        "year": year,
        "journal": journal,
        "volume": volume,
        "page": page,
        "cited_doi": cited_doi
    }

"""
PT 出版物类型,AU 作者,AF 作者全名,TI 文献标题,SO 出版物名称
LA 语种,DT 文献类型,CT 会议标题,CY 会议日期,CL 会议地点
DE 作者关键词,ID Keywords Plus®,AB 摘要,C1 作者地址
C3 作者机构,RP 通讯作者地址,EM 电子邮件地址,RI ResearcherID 号
OI ORCID 标识符,FU 基金资助机构和授权号,FX 基金资助正文,CR 引用的参考文献
NR 引用的参考文献数,TC Web of Science 核心合集的被引频次计数,Z9 被引频次合计
U1 使用次数（最近 180 天）,U2 使用次数（2013 年至今）,PU 出版商,PI 出版商所在城市
PA 出版商地址,SN 国际标准期刊号 (ISSN),EI 电子国际标准期刊号 (eISSN),J9 长度为 29 个字符的来源文献名称缩写
JI ISO 来源文献名称缩写,PD 出版日期,PY 出版年,VL 卷,IS 期
BP 开始页,EP 结束页,AR 文献编号,DI 数字对象标识符 (DOI),
PG 页数,WC Web of Science 类别,SC 研究方向,GA 文献传递号
UT 入藏号,PM PubMed ID,OA 公开访问指示符,DA 生成此报告的日期。
"""
# field_list=["PT","SO","LA","DT","CT","CY","CL","AB","C1","C3","RP",
#             "EM","RI","OI","FU","FX","PU","PI","PA","SN","EI","J9","JI",
#             "PD","PY","WC", "SC","GA","UT","PM","OA"]

field_list=["CT","CY","AB","PU","SN","EI","J9","JI","PY","WC","SC"]

def process_paper(tx, paper):
    current_id = paper.get("DI")
    # if(current_id !='empty'):
    #     tx.run(
    #         """
    #         MERGE (p:Paper {DI: $di})
    #         SET p.TI = $ti,p.NR = $nr,p.TC = $tc,p.Z9 = $z9,
    #             p.U1 = $u1,p.U2 = $u2,p.PG = $pg
    #         """,
    #         di=current_id,ti=paper["TI"],nr=paper["NR"],tc=paper["TC"],z9=paper["Z9"],
    #         u1=paper["U1"],u2=paper["U2"],pg=paper["PG"]
    #     )
    if (current_id != 'empty'):
        tx.run(
            """
            MERGE (p:Paper {DI: $di})
            SET p.TI = $ti
            """,
            di=current_id,
            ti=paper["TI"]
        )

    for element in field_list:
        if paper[element]!='empty':
            params ={"di":current_id,"value":paper[element]}
            tx.run(
                f"""
                MERGE (n:{element} {{{element.lower()}: $value}})
                WITH n
                MATCH (p:Paper {{DI: $di}})
                MERGE (p)-[r:{element}]->(n)
                """,
                **params
            )
    #AU
    # 处理作者关系
    for author in paper["AU"]:
        tx.run(
            """
            MERGE (a:Author {name: $name})
            WITH a
            MATCH (p:Paper {DI: $di})
            MERGE (p)-[:AU]->(a)
            """,
            name=author.strip(),
            di=current_id
        )

    #DE 关键词
    for kw in paper["keywords"]:
        tx.run(
            """
            MERGE (k:Keyword {keyword: $de})
            WITH k
            MATCH (p:Paper {DI: $di})
            MERGE (p)-[:KW]->(k)
            """,
            de=kw.strip(),
            di=current_id
        )
    # #ID kw plus
    # for kwp in paper["ID"]:
    #     if(kwp!='empty'):
    #         tx.run(
    #             """
    #             MERGE (i:ID {key_word_plus: $id_1})
    #             WITH i
    #             MATCH (p:Paper {DI: $di})
    #             MERGE (p)-[:ID]->(i)
    #             """,
    #             id_1=kwp.strip(),
    #             di=current_id
    #         )



    """第二部分"""
    # 处理引用关系
    for cr in paper.get("CR", []):
        cr_info = extract_doi(cr)

        if not cr_info:  # 跳过无法解析的条目
            continue
        cited_id = cr_info.get("cited_doi")
        if not cited_id:  # 跳过无 DOI 的引用
            continue

        if cr_info["volume"] is None:
            cr_info["volume"]="empty"
        if cr_info["page"] is None:
            cr_info["page"]="empty"
        #cited_id = cr_info["cited_doi"]

        if cited_id:
            # 创建被引用论文节点（若不存在）
            tx.run(
                """
                MERGE (p_cited:Paper {DI: $cited_id})
                ON CREATE SET p_cited.year = $year,
                              p_cited.journal = $journal,
                              p_cited.volume = $volume,
                              p_cited.page = $page
                """,
                cited_id=cited_id,
                year=cr_info["year"],
                journal=cr_info["journal"],
                volume=cr_info["volume"],
                page=cr_info["page"]
            )

            # 创建被引用论文节点的作者节点

            tx.run(
                """
                MERGE (a:Author {name: $name})
                WITH a
                MATCH (p:Paper {DI: $cite_id})
                MERGE (p)-[:AU]->(a)
                """,
                name=cr_info["author"],
                cite_id=cited_id
            )
            # 建立引用关系
            tx.run(
                """
                MATCH (p_citing:Paper {DI: $current_doi})
                MATCH (p_cited:Paper {DI: $cited_doi})
                MERGE (p_citing)-[:CITES]->(p_cited)
                """,
                current_doi=current_id,
                cited_doi=cited_id
            )

def main(json_path):
    with driver.session() as session:
        with open(json_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
            counter = 0
            for paper in papers:
                session.execute_write(process_paper, paper)
                counter += 1
                print("Processed {}/{}".format(counter, 5298))
    driver.close()

if __name__ == "__main__":
    # main("../data/jsondata/txtJsonData_clean_test.json")
    main("../data/jsondata/random_kw.json")