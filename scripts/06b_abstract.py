import re
path = r"../manuscript_draft.md"
text = open(path, encoding="utf-8").read()
m = re.search(r"## Abstract \(.*?\)\s*\n(.*?)\n\s*##", text, re.S)
if m:
    abstract = m.group(1)
    words = re.findall(r"[A-Za-z0-9]+", abstract)
    print("Abstract word count:", len(words))
else:
    print("not found")
