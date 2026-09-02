import re
path = r"../manuscript_draft.md"
text = open(path, encoding="utf-8").read()
m = re.search(r"## Abstract\s*\n(.*?)\n\s*##", text, re.S)
if m:
    abstract = m.group(1)
    words = re.findall(r"[A-Za-z0-9]+", abstract)
    print("Abstract word count:", len(words))
else:
    print("Abstract section not found")
# check remaining placeholders
ph = re.findall(r"\{\{.*?\}\}", text)
print("Remaining placeholders:", ph if ph else "NONE")
# word count of main body (after Abstract, before References)
m2 = re.search(r"## Abstract.*?\n\s*##\s*(.*)", text, re.S)
if m2:
    body = m2.group(1)
    ref_split = body.split("## References")
    main = ref_split[0]
    w = re.findall(r"[A-Za-z0-9]+", main)
    print("Main body word count (excl refs):", len(w))
print("Reference entries:", len(re.findall(r"^\d+\.\s", text, re.M)))
