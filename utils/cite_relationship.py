import json
import re

def extract_single_doi_from_cr(cr_entry):
    # 优先匹配标准 DOI 格式（如 DOI 10.xxx）
    match = re.search(r'.*?DOI\s+(?P<doi>10\.\d+/[^\s\],;]+)', cr_entry)
    if match:
        return match.group("doi").rstrip('/')
    else:
        return None

def find_citation_relations(data):
    doi_map = []
    results = [] # 存储返回结果
    for paper in data:
        doi_map.append(paper['DI'])
    for paper in data:
        if paper["DI"]=='empty':
            continue
        current_di = paper["DI"]
        cited_dois = []

        found = False
        for cr_entry in paper["CR"]:
            cited_doi = extract_single_doi_from_cr(cr_entry)
            if cited_doi and cited_doi in doi_map:
                cited_dois.append(cited_doi)
                found = True

        results.append({
            "doi": current_di,
            "cited_dois": cited_dois,
            "authors": paper["AU"],
            "keywords": paper["keywords"]
        })
        print(f"\nChecking citations for: {current_di}")
        if cited_dois:
            for doi in cited_dois:
                print(f" Cited: {doi}")
        else:
            print("  No internal citations found")
        print("=" * 35)

    return results

if __name__ == "__main__":
    with open("../data/jsondata/random_kw.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=== Simplified Citation Report ===")
    cite_res=find_citation_relations(data)

    output_path = "../data/jsondata/citation_relations.json"
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(cite_res, out_f, indent=2, ensure_ascii=False)
