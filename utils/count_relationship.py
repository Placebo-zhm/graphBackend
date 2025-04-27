import json
from collections import defaultdict

# Load paper citation data
with open('../data/jsondata/citation_relations.json', 'r') as f:
    papers = json.load(f)

doi_to_keywords = {paper['doi']: paper['keywords'] for paper in papers}

# Load keyword node data
with open('../data/jsondata/keyword_nodes.json', 'r') as f:
    nodes = json.load(f)

# Create node keyword set and map
node_keywords = {node['keyword_name'] for node in nodes}
keyword_dois_map = {node['keyword_name']: node['related_dois'] for node in nodes}

# Initialize result structure
result = {node['keyword_name']: {
    "keyword": node['keyword_name'],
    "cite_num": defaultdict(int)
} for node in nodes}

# Process each keyword
for current_kw in keyword_dois_map:
    # Get all papers related to current keyword
    for citing_doi in keyword_dois_map[current_kw]:
        # Skip if citing paper has no citation data
        if citing_doi not in doi_to_keywords:
            continue

        # Get cited papers by current paper
        cited_dois = next(
            (p['cited_dois'] for p in papers if p['doi'] == citing_doi),
            []
        )

        for cited_doi in cited_dois:
            # Skip if cited paper has no keywords
            if cited_doi not in doi_to_keywords:
                continue

            cited_kws = doi_to_keywords[cited_doi]
            # Exclude citations where current keyword exists in cited paper
            if current_kw in cited_kws:
                continue

            # Count valid keywords
            for cited_kw in cited_kws:
                if cited_kw in node_keywords:
                    result[current_kw]['cite_num'][cited_kw] += 1

# Convert defaultdict to regular dict and prepare output
output = []
for kw, data in result.items():
    output.append({
        "keyword": kw,
        "cite_num": dict(data['cite_num'])
    })

# Save results
with open('../data/jsondata/keyword_citation_count.json', 'w') as f:
    json.dump(output, f, indent=4)