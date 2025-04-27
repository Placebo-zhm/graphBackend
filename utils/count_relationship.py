import json
from collections import defaultdict

# Load paper citation data
with open('../data/jsondata/citation_relations.json', 'r') as f:
    papers = json.load(f)

doi_to_paper = {paper['doi']: paper for paper in papers}
doi_to_keywords = {paper['doi']: paper['keywords'] for paper in papers}

# Load keyword node data
with open('../data/jsondata/keyword_nodes.json', 'r') as f:
    nodes = json.load(f)

node_keywords = {node['keyword_name'] for node in nodes}

# Initialize result structure
result = {}
for node in nodes:
    keyword = node['keyword_name']
    result[keyword] = {
        "keyword": keyword,
        "cite_num": defaultdict(int)
    }

# Process each keyword's related papers
for node in nodes:
    current_keyword = node['keyword_name']
    related_dois = node['related_dois']

    for doi in related_dois:
        if doi not in doi_to_paper:
            continue  # Skip if paper not in citation data

        cited_dois = doi_to_paper[doi]['cited_dois']
        for cited_doi in cited_dois:
            if cited_doi not in doi_to_keywords:
                continue  # Skip if cited paper has no keyword data

            cited_keywords = doi_to_keywords[cited_doi]
            valid_keywords = [k for k in cited_keywords if k in node_keywords]

            for cited_kw in valid_keywords:
                result[current_keyword]['cite_num'][cited_kw] += 1

# Convert defaultdict to regular dict and prepare output
output = []
for keyword, data in result.items():
    cite_num = dict(data['cite_num'])
    output.append({
        "keyword": keyword,
        "cite_num": cite_num
    })

# Save results to JSON
with open('../data/jsondata/keyword_citation_count.json', 'w') as f:
    json.dump(output, f, indent=4)