#from p2neo4j import GraphDatabase
from neo4j import GraphDatabase, Neo4jConnection
import json
import re

# Neo4j连接配置
URI = "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = "12345678"

# 初始化驱动
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# 增强的DOI正则表达式（兼容更多格式）
#DOI_PATTERN = re.compile(r"DOI\s*:\s*([^\s,]+)|10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

# CR_PATTERN = re.compile(
#     r"""
#     ^
#     (?P<author>(.*?)\s*),\s*       # 作者（第一个逗号前的内容）
#     (?P<year>\d{4}),\s*         # 年份（4位数字）
#     (?P<journal>(.*?)\s*),\s*      # 期刊名称
#     (?:V(?P<volume>\d+),\s*)?   # 卷号（可选）
#     (?:P(?P<page>\d+),\s*)?     # 页码（可选）
#     (?:.*?DOI\s+(?P<doi>10\.[^\s,]+))  # DOI（必须存在）
#     """,
#     re.IGNORECASE | re.VERBOSE
# )
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


def process_paper(tx, paper):
    # 当前论文的唯一标识优先DOI
    # 流程介绍：1、获取paper的DI字段的doi，查询现有的引用论文节点，若没有则创建论文节点，若有则替换引文节点为论文节点，保持原有的关系
    #         2、
    #
    current_id = paper.get("DI")
    tx.run(
        """
        MERGE (p:Paper {id: $doi})
        SET p.TI = $ti,p.PT = $pt,p.SO = $so,p.LA = $la,p.DT = $dt,p.CT = $ct,
            p.CY = $cy,p.CL = $cl,p.DE = $de,p.ID = $id,p.AB = $ab,p.C1 = $c1,
            p.C3 = $c3,p.RP = $rp,p.EM = $em,p.RI = $ri,p.OI = $oi,p.FU = $fu,
            p.FX = $fx,p.NR = $nr,p.TC = $tc,p.Z9 = $z9,p.U1 = $u1,p.U2 = $u2,
            p.PU = $pu,p.PI = $pi,p.PA = $pa,p.SN = $sn,p.EI = $ei,p.J9 = $j9,
            p.JI = $ji,p.PD = $pd,p.VL = $vl,p.IS = $is_1,p.BP = $bp,p.EP = $ep,
            p.PG = $pg,p.WC = $wc,p.WE = $we,p.SC = $sc,p.GA = $ga,p.UT = $ut,
            p.PM = $pm,p.DA = $da
        """,
        doi=paper.get("DI"),ti=paper["TI"],pt=paper["PT"],so=paper["SO"],la=paper["LA"],
        dt=paper["DT"],ct=paper["CT"],cy=paper["CY"],cl=paper["CL"],de=paper["DE"],
        id=paper["ID"],ab=paper["AB"],c1=paper["C1"],c3=paper["C3"],rp=paper["RP"],
        em=paper["EM"],ri=paper["RI"],oi=paper["OI"],fu=paper["FU"],fx=paper["FX"],
        nr=paper["NR"],tc=paper["TC"],z9=paper["Z9"],u1=paper["U1"],u2=paper["U2"],
        pu=paper["PU"],pi=paper["PI"],pa=paper["PA"],sn=paper["SN"],ei=paper["EI"],
        j9=paper["J9"],ji=paper["JI"],pd=paper["PD"],py=paper["PY"],vl=paper["VL"],
        is_1=paper["IS"],bp=paper["BP"],ep=paper["EP"],pg=paper["PG"],wc=paper["WC"],
        we=paper["WE"],sc=paper["SC"],ga=paper["GA"],ut=paper["UT"],pm=paper["PM"],
        da=paper["DA"]
    )

    tx.run(
        """
        MERGE (p:Paper {id: $id})
        SET p.TI = $ti,
            p.SO = $so
        """,
        id=current_id,
        ti=paper["TI"],
        so=paper["SO"]
    )

    # 处理作者关系
    for author in paper["AU"]:
        tx.run(
            """
            MERGE (a:Author {name: $name})
            WITH a
            MATCH (p:Paper {id: $id})
            MERGE (a)-[:WROTE]->(p)
            """,
            name=author.strip(),
            id=current_id
        )

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
                MERGE (p_cited:Paper {id: $cited_id})
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
                MATCH (p:Paper {id: $cite_id})
                MERGE (a)-[:WROTE]->(p)
                """,
                name=cr_info["author"],
                cite_id=cited_id
            )

            # 建立引用关系
            tx.run(
                """
                MATCH (p_citing:Paper {id: $current_doi})
                MATCH (p_cited:Paper {id: $cited_doi})
                MERGE (p_citing)-[:CITES]->(p_cited)
                """,
                current_doi=current_id,
                cited_doi=cited_id
            )

def main(json_path):
    with driver.session() as session:
        with open(json_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
            for paper in papers:
                session.execute_write(process_paper, paper)
    driver.close()

if __name__ == "__main__":
    main("txtJsonData_clean.json")