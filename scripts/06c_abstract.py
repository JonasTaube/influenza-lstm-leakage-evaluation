import re
path = r"../manuscript_draft.md"
text = open(path, encoding="utf-8").read()
m = re.search(r"## Abstract \(.*?\)\s*\n(.*?)\n\s*\*\*Keywords", text, re.S)
if m:
    abstract = m.group(1).strip()
    tokens = abstract.split()
    print("Abstract word count (whitespace tokens):", len(tokens))
else:
    print("not found")
