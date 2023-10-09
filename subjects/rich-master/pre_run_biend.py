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
else:
    from tqdm import tqdm
# 5042845
# 72766257
# 4122233
# 18145034
from collections import defaultdict
import pickle as pkl
cnt = 0
nob = 0
ob = 0
nn = 0
ln = 0
rn = 0
modification = 0 
creation = 0 
extension = 0
type_changing_mod = 0 
cnts = defaultdict(int)


# left <: right 
# def is_subtype(left, right):
with open("ch", "rb") as f:
    class_hierarchy = pkl.load(f)

with open("file_funcs", "rb") as f:
    file_funcs = pkl.load(f)
# file_funcs = {}
with open("normal_variable_mutation", 'w+') as f:
    pass
with open("mutation_store_fast", 'w+') as f:
    pass

info_set = set()
def clear():
    global cnt, nob, ob, nn, info_set, mutative, cover, mutative_set, cover_set, ln, rn, modification, creation, extension, type_changing_mod
    cnt = 0
    nob = 0
    ob = 0
    nn = 0
    ln = 0
    rn = 0    
    mutative = 0
    cover = 0
    modification = 0 
    type_changing_mod = 0
    creation = 0 
    extension = 0
    mutative_set = set()
    cover_set = set()
    info_set = set()

def print_info():
    if cover > 0:
        print(f"mutative: {mutative}")
        print(f"portion: {mutative/cover}")
        print(f"portion set: {len(mutative_set)/len(cover_set)}")
    print(f"none none: {nn}")
    print(f"from none: {ln}")
    print(f"to none: {rn}")
    print(f"relational mutation: {nob}")
    print(f"normal mutation: {ob}")

    print(f"modification: {modification}")
    
    print(f"creation: {creation}")
    print(f"extension: {extension}")
    
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

dict_services = defaultdict(list)
object_reuses = defaultdict(int)
def change_obj(frame_name):
    if c != 'store_attr':
        return frame_name
    return '<obj>:' + frame_name.split(":")[1]
max_lines = 0

object_dicts = defaultdict(list)

def read_lines(path):

    lines = []
    with open(path, 'r', errors='ignore') as f:
        for line in tqdm(f):
            lines.append(line)
            if max_lines > 0 and len(lines) > max_lines:
                break
    return lines
def read_frames_biend(path):
    frames = []
    ok = defaultdict(int)
    cur = defaultdict(int)
    opcodes = defaultdict(list)
    infos = defaultdict(list)
    objects = defaultdict(list)
    serving = {}
    served = {}
    lines = read_lines(path)
    n = len(lines)

    for i in tqdm(range(n)):
        line = lines[i]
        if line.startswith("<opcode>"):
            linex = line.split("<opcode>")[1].strip()
            frame_name = ":".join(linex.strip().split(":")[:2])
            ok[frame_name] = 1
    for i in tqdm(range(n)):
        line = lines[i]
        if line.startswith("----------!"):
            frame_name = ":".join(line.split("----------!")[1].strip().split(":")[:2])
            frame_namex = change_obj(frame_name)
            if ok[frame_namex]:
                if c == 'store_attr':
                    short_master = frame_name.split(":")[0].split('-')[1]
                else:
                    short_master = frame_name.split(":")[0]
                master = frame_name.split(":")[0] + ':' + str(object_reuses[short_master])
                servant = frame_namex
                # serving[frame_namex] = frame_name.split(":")[0] + ':' + str(object_reuses[frame_name.split(":")[0]])
                serving[servant] = master
                served[short_master] = servant
                if c == 'store_attr':
                    dict_services[servant].append(master)
                opcodes[frame_namex].append([])
                infos[frame_namex].append([])
        elif line.startswith("----------^"):
            frame_name = ":".join(line.split("----------^")[1].strip().split(":")[:2])

            frame_namex = change_obj(frame_name)
            if ok[frame_namex]:
                if c == 'store_attr':
                    served[serving[frame_namex]] = None
                    serving[frame_namex] = None

        elif line.startswith("----------~"):
            frame_name = ":".join(line.split("----------~")[1].strip().split(":")[:2])
            frame_namex = change_obj(frame_name)

            if c == 'store_attr':

                short_master = frame_name.split(":")[0].split('-')[1]
                if short_master in served and served[short_master] != None:
                    serving[served[short_master]] = None
                    object_reuses[short_master] += 1
            elif c == 'dict':
                if ok[frame_namex]:
                    serving[frame_namex] = None
        elif line.startswith("<opcode>"):
            linex = line.split("<opcode>")[1].strip()
            frame_name = ":".join(linex.strip().split(":")[:2])
            if frame_name in opcodes and frame_name in serving and serving[frame_name] is not None:
                opcodes[frame_name][-1].append(line)
            # if c == 'store_attr':
            #     if frame_name in opcodes and frame_name in serving and serving[frame_name] is not None:
            #         opcodes[frame_name][-1].append(line)
            # else:
            #     if frame_name in opcodes and len(opcodes[frame_name]) > 0:
            #         opcodes[frame_name][-1].append(line)
        else:
            frame_name = ":".join(line.strip().split(":")[:2])
            if frame_name in infos and frame_name in serving and serving[frame_name] is not None:
                infos[frame_name][-1].append(line.strip())
            # if c == 'store_attr':
            #     if frame_name in infos and frame_name in serving and serving[frame_name] is not None:
            #         infos[frame_name][-1].append(line.strip())
            # else:
            #     if frame_name in opcodes and len(opcodes[frame_name]) > 0:
            #         infos[frame_name][-1].append(line.strip())
    for frame_name in tqdm(opcodes):
        if c == 'store_attr':
            assert len(opcodes[frame_name]) == len(dict_services[frame_name])
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
                        if ((idx.find(repo) != -1 or idx.find("<string>: __init__") != -1 ) and info.find(info_excludes[0]) == -1):
                            # good frame node
                            frame += ((idx, info),)
                            check_info_while_reading(idx, info)
                    # if frame not in seen:
                    #     # good frame
                    #     seen.add(frame)
                    if len(frame) > 0:
                        frames.append(frame)
                    if c == 'store_attr':
                        object_dicts[dict_services[frame_name][i]].append(frame)
    return frames


def read_frames(path):
    frames = []
    ok = defaultdict(int)
    cur = defaultdict(int)
    opcodes = defaultdict(list)
    infos = defaultdict(list)
    lines = read_lines(path)
    n = len(lines)
    for i in tqdm(range(n)):
        line = lines[i]
        if line.startswith("<opcode>"):
            linex = line.split("<opcode>")[1].strip()
            frame_name = ":".join(linex.strip().split(":")[:2])
            ok[frame_name] = 1
    for i in tqdm(range(n)):
        line = lines[i]
        if line.startswith("----------"):
            frame_name = ":".join(line.split("----------")[1].strip().split(":")[:2])
            if ok[frame_name]:
                opcodes[frame_name].append([])
                infos[frame_name].append([])
    for i in tqdm(range(n)):
        line = lines[i]
        if line.startswith("----------"):
            frame_name = ":".join(line.split("----------")[1].strip().split(":")[:2])
            if ok[frame_name]:
                cur[frame_name] += 1
        elif line.startswith("<opcode>"):
            linex = line.split("<opcode>")[1].strip()
            frame_name = ":".join(linex.strip().split(":")[:2])
            if frame_name in opcodes and cur[frame_name] < len(opcodes[frame_name]):
                opcodes[frame_name][cur[frame_name]-1].append(line)
        else:
            frame_name = ":".join(line.strip().split(":")[:2])
            if frame_name in infos and cur[frame_name] < len(infos[frame_name]):
                infos[frame_name][cur[frame_name]-1].append(line.strip())
    for frame_name in tqdm(opcodes):
        if len(opcodes[frame_name]) == len(infos[frame_name]):
            n = len(opcodes[frame_name])
            seen = set()
            for i in range(n):
                if len(opcodes[frame_name][i]) == len(infos[frame_name][i]):
                    m = len(opcodes[frame_name][i])
                    # frame = tuple() # if we do not want to de-replicate, we can use list, which append in O(1) time. 
                    frame = []
                    for j in range(m):
                        idx = opcodes[frame_name][i][j]
                        info = infos[frame_name][i][j]
                        if ((idx.find(repo) != -1 or idx.find("<string>: __init__") != -1 ) and info.find(info_excludes[0]) == -1):
                            # good frame node
                            frame.append((idx, info))
                            # check_info_while_reading(idx, info)
                    # if frame not in seen:
                    #     # good frame
                    #     seen.add(frame)
                    if len(frame) > 0:
                        frames.append(frame)
    return frames


def remove_frame_addr(idx, frame_id):
    new_idx = idx.split(":")[2:]
    new_idx[frame_id] = idx.split(":")[frame_id][:-15]
    return ":".join(new_idx)
def _analyze_fast(frame):
    global cnt, nn, rn, ln, nob, ob, cover, mutative, cnts
    vlc = defaultdict(list)
    vt = defaultdict(set)
    type_env = {}
    for idx, info in frame:
        with open("bound_events", 'a+') as f:
            f.write(idx)
            f.write(info+'\n')
        cover += 1
        cover_set.add(remove_frame_addr(idx, 1))
        msg = info.split(":")
        name, t = msg[2].strip(), msg[3].strip()
        if name in type_env and idx.find('config') == -1:
            if t != type_env[name] and remove_frame_addr(idx, 1) not in mutative_set:
                mutative += 1
                mutative_set.add(remove_frame_addr(idx, 1))
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
                    if type_env[name] == 'builtins.NoneType' and t == 'builtins.NoneType':
                        nn += 1
                    elif type_env[name] == 'builtins.NoneType':
                        ln += 1
                    else:
                        rn += 1
        type_env[name] = t
        vt[name].add(t)
        # name, from_t, to_t = msg[2].strip(), msg[3].strip(), msg[4].strip()
        # vlc[name].append((from_t, to_t))
        # vt[name].add(from_t)
        # vt[name].add(to_t)
    for v in vt:
        cnts[len(vt[v])] += 1



def analyze_attr_diff(frame):
    global nn, ln, rn, nob, ob, mutative, cover, modification, creation, extension, type_changing_mod
    for idx, info in frame:
        cover += 1
        cover_set.add(remove_frame_addr(idx, 1))
        msg = info.split(":")
        if len(msg) == 8:
            mutative += 1
            from_t, t = msg[6].strip(), msg[7].strip()
            if from_t != 'builtins.NoneType' and t != 'builtins.NoneType':
                print(idx)
                print(info)
                common_superclasss = elca(from_t, t)
                if len(common_superclasss) > 1:
                    nob += 1
                else:
                    ob += 1
                    # with open(f"mutation_store_attr", 'a+') as f:
                    #     f.write(idx)
                    #     f.write(info+'\n')
            else:
                if from_t == 'builtins.NoneType' and t == 'builtins.NoneType':
                    nn += 1
                elif from_t == 'builtins.NoneType':
                    ln += 1
                else:
                    rn += 1

def _analyze_attr(frame):
    global nn, ln, rn, nob, ob, mutative, cover, modification, creation, extension, type_changing_mod, class_objects, obj, ext, extensive_class
    vt = {}
    extensive = 0
    # type_env = defaultdict(dict)
    type_env = {}
    vlc = defaultdict(dict)
    for idx, info in frame:
        cover += 1
        idx3 = remove_frame_addr(idx, 1)
        cover_set.add(idx3)
        msg = info.split(":")
        if len(msg) == 7:
            name, attr, t = msg[1].strip(), msg[4].strip(), msg[5].strip()
            if attr in type_env:
                # modification
                modification += 1
                if t != type_env[attr]:
                    mutative += 1
                    if idx3 not in mutative_set or True:
                        mutative_set.add(idx3)
                        if type_env[attr] != 'builtins.NoneType' and t != 'builtins.NoneType':
                            # print(idx)
                            # print(info)
                            common_superclasss = elca(type_env[attr], t)
                            if len(common_superclasss) > 1:
                                nob += 1
                            else:
                                # float, int
                                # renderable
                                ob += 1
                                with open(f"mutation_store_attr", 'a+') as f:
                                    f.write(idx)
                                    f.write(info+'\n')
                        else:
                            if type_env[attr] == 'builtins.NoneType' and t == 'builtins.NoneType':
                                nn += 1
                            elif type_env[attr] == 'builtins.NoneType':
                                ln += 1
                            else:
                                rn += 1
            else:
                func = msg[3].strip()[:12]
                if func == 'self-initing' or func.find('copy') != -1 or idx.find('new') != -1:
                    creation += 1
                else:
                    if idx3 not in idxs:
                        idxs.add(idx3)
                        extension += 1
                        extensive = 1
            type_env[attr] = t
            if attr in vt:    
                vt[attr].append(t)
            else:
                vt[attr] = [t]
        else:
            # assert False
            pass

    if extensive:
        ext += 1
        extensive_class.add(obj.split(":")[0].split('-')[0])
    class_objects[obj.split(":")[0].split('-')[0]].add(frozenset(type_env.items()))
            
def only_type(keys):
    return {k.split("-")[1] for k in keys}


def _analyze_dict(frame):
    global nn, ln, rn, nob, ob, mutative, cover, modification, creation, extension, type_changing_mod
    global only_create, not_only_create, hete_value, hete_key, delete_dict, clear_dict
    vt = defaultdict(dict)
    type_env = defaultdict(dict)

    vlc = defaultdict(dict)
    has_delete = False
    has_clear = False
    lines = []
    file = None
    for idx, info in frame:
        cover_set.add(remove_frame_addr(idx, 1))
        msg = info.split(":")
        file = idx.split(":")[2].strip()
        lines.append(idx.split(":")[4].strip())
        if len(msg) == 7:
            # insert

            cover += 1
            key, value = msg[4].strip(), msg[5].strip()
            if key in type_env:
                if type_env[key] != value:
                    mutative += 1
            type_env[key] = value
        elif len(msg) == 6:
            # delete
            key = msg[4].strip()
            if key in type_env:
                del type_env[key]
            has_delete = True
        elif len(msg) == 5:
            # clear
            type_env.clear()
            has_clear = True
    if file.find('tests') == -1  and not (len(set(lines)) == 1 and file in file_funcs and int(lines[0]) in file_funcs[file]):
        line_tuple = tuple(set(lines))
        if line_tuple not in line_cache:
            line_cache.add(line_tuple)
            if len(set(type_env.values())) > 1:
                hete_value += 1
            if len(set(only_type(type_env.keys()))) > 1:
                hete_key += 1
            if len(set(lines)) == 1:
                only_create += 1
            else:
                not_only_create += 1
            if has_delete:
                delete_dict += 1
            if has_clear:
                clear_dict += 1

def analyze_fast():
    print("----------store_fast----------")
    for frame in tqdm(frames["store_fast"]):
        _analyze_fast(frame)
    print_info()
    print(cnts)
    clear()

idxs = set()
class_objects = defaultdict(set)
mul = 0
sig = 0
poly_cls = 0
obj = None
ext = 0
extensive_class = set()
def analyze_attr():
    global poly_cls, sig, mul, class_objects, idxs, obj, ext, extensive_class
    print("----------store_attr----------")
    # merge frames since object may occur in different frames
    for obj in object_dicts:
        if obj.find('(nil):0') != -1:
            continue
        if obj.find('builtins') != -1:
            continue
        
        if len(object_dicts[obj]) > 1:
            # dict changes
            mul += 1
        else:
            sig += 1
        for frame in object_dicts[obj]:
            _analyze_attr(frame)
    print_info()
    for clas in class_objects:
        # poly class很多，说明class不适合做summarizing
        if len(class_objects[clas]) > 1:
            poly_cls += 1

    print('extensive_object:' + str(ext/len(object_dicts)))
    print('multiple dict: ' + str(mul))
    print('single dict: ' + str(sig))
    print('polymorphic class: ' + str(poly_cls))
    clear()

only_create = 0
not_only_create = 0
line_cache = set()
hete_value = 0 
hete_key = 0
delete_dict = 0
clear_dict = 0
def analyze_dict():
    print("----------store_dict----------")
    # merge frames since object may occur in different frames

    for frame in tqdm(frames["dict"]):
        _analyze_dict(frame)
    print_info()
    print('only_create: ' + str(only_create))
    print('not_only_create: ' + str(not_only_create))
    print('hete_value: ' + str(hete_value))
    print('hete_key: ' + str(hete_key))
    print('delete_dict: ' + str(delete_dict))
    print('clear_dict: ' + str(clear_dict))

frames = {"store_fast": defaultdict(list), "store_name": defaultdict(list), "store_global": defaultdict(list), "store_attr": defaultdict(list)}
analyzers = {"store_fast": analyze_fast, "store_attr": analyze_attr, "dict": analyze_dict}

for c in files:
    opcodes = defaultdict(list)
    infos = defaultdict(list)
    mutative = 0
    cover = 0
    mutative_set = set()
    cover_set = set()
    if c == 'store_fast':
        f = read_frames(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flow.txt")
    else:    
        f = read_frames_biend(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flow.txt")
    frames[c] = f
    analyzers[c]()



