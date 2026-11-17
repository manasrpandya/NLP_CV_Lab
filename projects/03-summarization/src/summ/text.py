import re

# plain regex sentence splitter, no nltk data needed (the demo page uses the same rule in js)
_sp = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")
_abbr = re.compile(r"\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|u\.s|u\.k)\.$", re.I)


def sents(x):
    x = re.sub(r"\s+", " ", x).strip()
    out, cur = [], ""
    for part in _sp.split(x):
        cur = f"{cur} {part}".strip() if cur else part
        if _abbr.search(cur) or len(cur.split()) < 3:
            continue
        out.append(cur)
        cur = ""
    if cur:
        if out:
            out[-1] = f"{out[-1]} {cur}"
        else:
            out.append(cur)
    return out


def words(x):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", x.lower())
