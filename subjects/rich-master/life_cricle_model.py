
from collections import defaultdict
import pickle as pkl
cnt = 0
nob = 0
ob = 0
nn = 0
cnts = defaultdict(int)
# left <: right 
# def is_subtype(left, right):


# not actually c3 for now. 
def mro(x) -> set:
    if x == "builtins.object":
        return {x}
    if x not in class_hierarchy:
        return {x, "builtins.object"}
    r = {x}
    for s in class_hierarchy[x]:
        r.update(mro(s))
    return r


def elca(a, b):
    global nob, ob
    # get the mro of a
    mro_a = mro(a)
    mro_b = mro(b)
    # get the mro of b
    common_superclasss = mro_a.intersection(mro_b)
    if len(common_superclasss) > 1:
        nob += 1
    else:
        ob += 1
def analyze(frame):
    global cnt, nn
    vlc = defaultdict(list)
    vt = defaultdict(set)
    assert len(frame) % 2 == 0
    n = len(frame)
    for i in range(0, n, 2):
        idx = frame[i]
        info = frame[i+1].strip()
        msg = info.split(":")
        name, from_t, to_t = msg[2].strip(), msg[3].strip(), msg[4].strip()
        
        vlc[name].append((from_t, to_t))
        if from_t != 'builtins.NoneType' and to_t != 'builtins.NoneType':
            elca(from_t, to_t)
            nn += 1
        else:
            pass
        vt[name].add(from_t)
        vt[name].add(to_t)

    for v in vt:
        cnts[len(vt[v])] += 1
previous = []
with open("ch", "rb") as f:
    class_hierarchy = pkl.load(f)
with open('store_fast_simp.txt') as f:
    for line in f:
        if line.startswith("----------"):
            analyze(previous)
            previous = []
        else:
            previous.append(line)

# print(cnt)
print(nn)
print(nob)
print(ob)
print(cnts)