from re import L

import sys
from collections import defaultdict
# 
# files = ['store_fast', 'store_name', 'store_global', 'store_attr']
files = ['store_fast', 'store_attr', 'dict']

repo = sys.argv[1]
only_analysis = sys.argv[2]
info_excludes = ['@py_assert']

if not int(only_analysis):
    for c in files:
        with open(f"/home/user/purepython/cpython-3.9/pydyna/{c}.txt", 'w+') as f:
            pass
        with open(f"/home/user/purepython/cpython-3.9/pydyna/{c}_cov.txt", 'w+') as f:
            pass
        with open(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flow.txt", 'w+') as f:
            pass
    import pytest
    pytest.main(['./tests'])


from collections import defaultdict
import pickle as pkl
cnt = 0
nob = 0
ob = 0
nn = 0
cnts = defaultdict(int)
# left <: right 
# def is_subtype(left, right):
with open("ch", "rb") as f:
    class_hierarchy = pkl.load(f)

with open("normal_variable_mutation", 'w+') as f:
    pass
info_set = set()
def clear():
    global cnt, nob, ob, nn, info_set, mutative, cover, mutative_set, cover_set
    cnt = 0
    nob = 0
    ob = 0
    nn = 0    
    mutative = 0
    cover = 0
    mutative_set = set()
    cover_set = set()
    info_set = set()

def print_info():
    
    if cover > 0:
        print(f"mutative: {mutative}")
        print(f"portion: {mutative/cover}")
        print(f"portion set: {len(mutative_set)/len(cover_set)}")
    print(f"none mutation: {nn}")
    print(f"relational mutation: {nob}")
    print(f"normal mutation: {ob}")
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
    return common_superclasss

def mutate_fast(info):
    global nob, ob, nn, info_set
    filtered = 0
    msg = info.split(":")
    name, from_t, to_t = msg[2].strip(), msg[3].strip(), msg[4].strip()
    if from_t != 'builtins.NoneType' and to_t != 'builtins.NoneType':
        common_superclasss = elca(from_t, to_t)
        if len(common_superclasss) > 1:
            nob += 1
        else:
            ob += 1
            with open("normal_variable_mutation", 'a+') as f:
                f.write(idx)
                f.write(info+'\n')
                filtered = 1
    else:
        nn += 1

def mutate_attr(info):
    global nob, ob, nn
    msg = info.split(":")
    if len(msg) == 5:
        name, from_t, to_t = msg[2].strip(), msg[3].strip(), msg[4].strip()
        if from_t != 'builtins.NoneType' and to_t != 'builtins.NoneType':
            common_superclasss = elca(from_t, to_t)
            if len(common_superclasss) > 1:
                nob += 1
            else:
                ob += 1
        else:
            nn += 1
    elif len(msg) == 4:
        assigned_t = msg[3].strip()
def mutate(info):

    global c
    if c == "store_fast":
        mutate_fast(info)
    elif c == "store_attr":
        mutate_attr(info)
    
def check_info_while_reading(idx, info):
    pass
max_lines = 0 
def read_frames(path):
    frames = []

    opcodes = defaultdict(list)
    infos = defaultdict(list)
    with open(path, 'r') as f:
        if max_lines:
            lines = []
            for line in f:
                lines.append(line)
                if len(lines) > max_lines:
                    break
        else:
            lines = f.readlines()
        n = len(lines)
        for i in range(n):
            line = lines[i]
            if line.startswith("----------"):
                frame_name = ":".join(line.split("----------")[1].strip().split(":")[:2])
                if len(opcodes[frame_name]) == 0:
                    opcodes[frame_name].append([])
                    infos[frame_name].append([])
        
        for i in range(n):
            line = lines[i]
            if line.startswith("----------"):
                frame_name = ":".join(line.split("----------")[1].strip().split(":")[:2])
                opcodes[frame_name].append([])
                infos[frame_name].append([])
            elif line.startswith("<opcode>"):
                linex = line.split("<opcode>")[1].strip()
                frame_name = ":".join(linex.strip().split(":")[:2])
                if frame_name in opcodes and len(opcodes[frame_name]) > 0:
                    opcodes[frame_name][-1].append(line)
            else:
                frame_name = ":".join(line.strip().split(":")[:2])
                if frame_name in infos and len(infos[frame_name]) > 0:
                    infos[frame_name][-1].append(line.strip())
        for frame_name in opcodes:
            if len(opcodes[frame_name]) == len(infos[frame_name]):
                n = len(opcodes[frame_name])
                seen = set()
                for i in range(n):
                    if len(opcodes[frame_name][i]) == len(infos[frame_name][i]):
                        m = len(opcodes[frame_name][i])
                        frame = tuple()
                        for j in range(m):
                            idx = opcodes[frame_name][i][j]
                            info = infos[frame_name][i][j]
                            if (idx.find(repo) != -1 and info.find(info_excludes[0]) == -1):
                                # good frame node
                                frame += ((idx, info),)
                                check_info_while_reading(idx, info)
                        if frame not in seen:
                            # good frame
                            seen.add(frame)
                            frames.append(frame)
    return frames
frames = {"store_fast": defaultdict(list), "store_name": defaultdict(list), "store_global": defaultdict(list), "store_attr": defaultdict(list)}
for c in files:
    opcodes = defaultdict(list)
    infos = defaultdict(list)
    mutative = 0
    cover = 0
    mutative_set = set()
    cover_set = set()
    f = read_frames(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flow.txt")
    frames[c] = f
    # if c == 'store_name':
    #     continue
    # with open(f"/home/user/purepython/cpython-3.9/pydyna/{c}_cov.txt", 'r') as f:

    #     lines = f.readlines()
    #     n = len(lines)
    #     for i in range(n):
    #         line = lines[i]
    #         if line.startswith("----------"):
    #             frame_name = ":".join(line.split("----------")[1].strip().split(":")[:2])
    #             if len(opcodes[frame_name]) == 0:
    #                 opcodes[frame_name].append([])
    #                 infos[frame_name].append([])
        
    #     for i in range(n):
    #         line = lines[i]
    #         if line.startswith("----------"):
    #             frame_name = ":".join(line.split("----------")[1].strip().split(":")[:2])
    #             opcodes[frame_name].append([])
    #             infos[frame_name].append([])
    #         elif line.startswith("<opcode>"):
    #             line = line.split("<opcode>")[1].strip()
    #             frame_name = ":".join(line.strip().split(":")[:2])
    #             opcodes[frame_name][-1].append(line)
    #         else:
    #             frame_name = ":".join(line.strip().split(":")[:2])
    #             infos[frame_name][-1].append(line.strip())
    #     for frame_name in opcodes:
    #         assert len(opcodes[frame_name]) == len(infos[frame_name])
    #         n = len(opcodes[frame_name])
    #         for i in range(n):
    #             assert len(opcodes[frame_name][i]) == len(infos[frame_name][i])
    #             m = len(opcodes[frame_name][i])
    #             for j in range(m):
    #                 idx = opcodes[frame_name][i][j]
    #                 info = infos[frame_name][i][j]
    #                 if (idx.find(repo) != -1 and info.find(info_excludes[0]) == -1):
    #                     # good frame node
    #                     cover += 1
    #                     cover_set.add(idx)
    
    print(c)
    if cover > 0:
        print(mutative/cover)
        print(len(mutative_set)/len(cover_set))
    else:
        print("cover is 0")
    
    print('-----------------')
    print_info()
    clear()

def analyze_fast(frame):
    global cnt, nn, nob, ob, cover, mutative, cnts
    vlc = defaultdict(list)
    vt = defaultdict(set)
    type_env = {}
    for idx, info in frame:
        cover += 1
        cover_set.add(idx)
        msg = info.split(":")
        name, t = msg[2].strip(), msg[3].strip()
        if name in type_env:
            if t != type_env[name]:
                mutative += 1
                mutative_set.add(idx)
                if type_env[name] != 'builtins.NoneType' and t != 'builtins.NoneType':
                    common_superclasss = elca(type_env[name], t)
                    if len(common_superclasss) > 1:
                        nob += 1
                    else:
                        ob += 1
                        with open("mutation_store_fast", 'a+') as f:
                            f.write(idx)
                            f.write(info+'\n')
                else:
                    nn += 1
        type_env[name] = t
        vt[name].add(t)
        # name, from_t, to_t = msg[2].strip(), msg[3].strip(), msg[4].strip()
        # vlc[name].append((from_t, to_t))
        # vt[name].add(from_t)
        # vt[name].add(to_t)
    for v in vt:
        cnts[len(vt[v])] += 1


def analyze_attr(frame):
    global nn, nob, ob, mutative, cover
    for idx, info in frame:
        cover += 1
        cover_set.add(idx)
        msg = info.split(":")
        if len(msg) == 6:
            name, attr, from_t, to_t = msg[2].strip(), msg[3].strip(), msg[4].strip(), msg[5].strip()
            if attr in vlc[name]:
                vlc[name][attr].append((from_t, to_t))
            else:
                vlc[name][attr] = [(from_t, to_t)]
            if attr in vt[name]:
                vt[name][attr].add(from_t)
                vt[name][attr].add(to_t)
            else:
                vt[name][attr] = {from_t, to_t}
            
        elif len(msg) == 5:
            name, attr, t = msg[2].strip(), msg[3].strip(), msg[4].strip()


            if attr in type_env[name]:
                if t != type_env[name][attr]:
                    mutative_set.add(idx)
                    mutative += 1
                    if type_env[name][attr] != 'builtins.NoneType' and t != 'builtins.NoneType':
                        common_superclasss = elca(type_env[name][attr], t)
                        if len(common_superclasss) > 1:
                            nob += 1
                        else:
                            ob += 1
                            with open(f"mutation_store_attr", 'a+') as f:
                                f.write(idx)
                                f.write(info+'\n')
                    else:
                        nn += 1
            type_env[name][attr] = t
            if attr in vt[name]:
                vt[name][attr].append(t)
            else:
                vt[name][attr] = [t]
            

print("----------store_fast----------")
for frame in frames["store_fast"]:
    analyze_fast(frame)
print_info()
print(cnts)
clear()
print("----------store_attr----------")
# merge frames since object may occur in different frames
vt = defaultdict(dict)
type_env = defaultdict(dict)

vlc = defaultdict(dict)

for frame in frames["store_attr"]:
        analyze_attr(frame)
print_info()
