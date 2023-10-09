from decimal import localcontext
from email.policy import default
from math import ceil, floor
from ossaudiodev import error
from statistics import variance
import sys
from collections import defaultdict
import os
from tokenize import group
import time
import json

GENERATE_FIG = True 
ONLY_NONIMAL = True
USE_SCI = False
USE_PERC = True
ONLY_CLASS_ANALYSIS = False
FRESH_ANNOTATION = False
SAVE_NOMINAL = False
LOC_K = 4
# 
# files = ['store_fast', 'store_name', 'store_global', 'store_attr']
files = ['store_attr']
# pyod, seab
fast_repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    'newspaper', 'impacket', 'routersploit', 'pre_commit', 'jinja', 'pendulum', 'wordcloud', 'pinyin', 'arrow', 'nltk', 
    'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream', 'pyod', 'altair', 'snorkel']
repo = sys.argv[1]
only_analysis = int(sys.argv[2])
fresh = int(sys.argv[3])

FRESH_INTE = int(sys.argv[4])
info_excludes = ['@py_assert']
global_class_classes = {}
import sys
POSSIBLE_FUNC = ['builtins.function', 'builtins.staticmethod', 'builtins.method']
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if not int(only_analysis):
    # for c in files:
    #     with open(f"/home/user/purepython/cpython-3.9/pydyna/{c}.txt", 'w+') as f:
    #         pass
    #     with open(f"/home/user/purepython/cpython-3.9/pydyna/{c}_cov.txt", 'w+') as f:
    #         pass
    #     with open(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flow.txt", 'w+') as f:
    #         pass
    import pytest
    # pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:cov', '-p', 'no:html', '-p', 'no:timeout', '-p', 'no:httpbin', '-p', 'no:hypothesis', '-p', 'no:mock', '-p', 'no:metadata', './tests/'])
    if repo == 'newspaper':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', './tests/unit_tests.py'])
    elif repo == 'impacket':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', './tests/ImpactPacket/'])
    elif repo == 'torch':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', './test/test_linalg.py', './test/test_tensorexpr.py', 'test/test_optim.py'])
    elif repo == 'pyod':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', './test/test_vae.py', 'test/test_kpca.py', 'test/test_knn.py', 'test/test_auto_encoder.py'])
    elif repo == 'statsmodels':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', 'regression/tests/test_regression.py', 'distributions/tests/', 'formula/tests/', 'stats/tests/'])
    elif repo == 'sphinx':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', 'tests/'])
    elif repo == 'spacy':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', 'tests/vocab_vectors/', 'tests/parser/', 'tests/tokenizer/'])
    elif repo == 'featuretools':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', 'tests/computational_backend', 'tests/entityset_tests'])
    elif repo == 'librosa':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', 'tests/test_core.py'])
    elif repo == 'pyro':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', './tests/distributions/'])
    elif repo in ['kornia', 'nltk', 'beets', 'wordcloud', 'snorkel', 'jedi']:
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', './test/'])
    elif repo == 'kombu':
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', './t'])
    else:    
        pytest.main(['-p', 'no:nbval', '-p', 'no:Faker', '-p', 'no:asyncio', '-p', 'no:timeout', '-p', 'no:metadata', './tests/'])
    
    # pytest.main(['./tests/'])
    
else:
    from tqdm import tqdm
# 5042845
# 72766257
# 4122233
# 18145034
from collections import defaultdict
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
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

from lark import Lark
from lark.tree import Tree
from lark.exceptions import UnexpectedCharacters
import random
import ast
import pprint2
from wp_intra_simp import Analyzer, Simplifier, cnf_expr, _expand
object_parser = Lark(open("object.lark").read(), start="t")
dict_parser = Lark(open("dict.lark").read(), start="dict")
tuple_parser = Lark(open("tuple.lark").read(), start="tuple")
mro_parser = Lark(open("mro.lark").read(), start="mro")
s = Lark(open("4.2.3.lark").read(), start="s")


id_to_master = {}
J1 = 0
J1s = defaultdict(int)
J1_ = 0
J1_s = defaultdict(int)
J2 = 0
J2s = defaultdict(int)
J2_ = 0
J2_s = defaultdict(int)

ASS = set()
E = []
D = []
C = []
R = []
REFL = []
SL1 = []
SL2 = []
SL3 = []
SL4 = []
SL5 = []

oom_names = ['set-descr', 'extend', 'override', 'update', 'get-descr', 'inherit', 'del-descr', 'remove', 'abnormal', 'abmonotonic', 'abextend', 'evolve', 'HYB', 'EXT', 'TYP', 'object_func', 'both', 'only-e', 'only-u', 'inlinear-value']

OM = oom_names # ['extend', 'override', 'update']
ALL_OOMs = {k:[] for k in oom_names}
ALL_OOM = {k:[0,0,0,0] for k in OM}

SSS1 = SSS2 = SSS3 = SSS4 = SSS5 =SSS6 = SSS7 = SSL1 = SSL2 = SSL3 = SSL4 = SSL5 = SSL6 = SSL7 = 0
LSS1 = LSS2 = LSS3 = LSL1 = LSL2 = LSL3 = 0
DSS1 = DSS2 = DSL1 = DSL2 = 0

LL1 = []
LL2 = []
LL3 = []
LD = defaultdict(list)
SD = defaultdict(list)


DELETE_INFOS = set()

M = []
U = []
F = defaultdict(int)
F2 = defaultdict(int)

S = set()
M1 = []
M2 = []
M3 = []
M4 = []
M5 = []
M6 = []
M7 = []
M8 = []

PAT = defaultdict(set)
PAT2 = defaultdict(int)

YES = 0
YES_SITE = set()
GUARDED = 0
GUARDED_SITE = set()
GUARDED_SITE_AGGRE = set()
REFINE_SITE = set()
UNION_SITE = set()
CONMIS = set()
MIS = set()
POLY_ATTR0 = set()
YES_SITE0 = set()
NOMINAL_ATTR0 = set()
GUARDED_SITE0 = set()
UNION_SITE0 = set()
REFINE_SITE0 = set()

SENSI_SITE1 = [set() for i in range(LOC_K)]
SENSI_SITE2 = set()
SENSI_SITE3 = set()
SENSI_SITE4 = set()

CAUSE_ATTR0 = set()
CAUSE_ATTR1 = set()
CAUSE_ATTR2 = set()

NOMINAL_ATTR = set()

POLY = 0
POLY_ATTR = set()
ALL_ATTR = set()
UPD = 0
NON_UPD = 0

OVR = 0
NON_OVR = 0
POLY_CLS = 0
METH_POLY_CLS = 0
RR = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'black', 'networkx', 'click', 'pydantic', 
    'yapf', 'faker', 'sklearn', 'pandas', 'mypy', 'pelican', 'newspaper', 'impacket', 'routersploit', 'pre_commit']
inherited = defaultdict(int)
inherited_descr = defaultdict(int)

inherited_repo = {k: defaultdict(int) for k in RR}

M1ov = []
M2ov = []
M3ov = []
M4ov = []
M5ov = []
M6ov = []
M7ov = []
M8ov = []
con_objs = [defaultdict(dict) for i in range(LOC_K)]
class_objs = defaultdict(dict)
POLY_CONS = 0
POLY_CONS2 = 0
POLY_HYB = 0
STA_CLS = 0
POLY_CONS_HYB = 0
POLY_CONS_EXT = 0
POLY_CONS_TYP = 0

POLY_CONS_HYB2 = 0
POLY_CONS_EXT2 = 0
POLY_CONS_TYP2 = 0

POLY_CONS_PARA = 0
POLY_CONS_LOC = [0 for i in range(LOC_K)]
ALLCOND = set()
UNCOND = set()
HASCOND = set()
OTHERCOND1 = set()
OTHERCOND2 = set()
OTHERCOND3 = set()

SIM_ATTR = 0
STRU_ATTR = 0
COER_ATTR = 0 
NATTR = 0
EATTR = 0
OTHER_ATTR = 0

SIM_ATTR2 = 0
STRU_ATTR2 = 0
COER_ATTR2 = 0 
EATTR2 = 0
NATTR2 = 0
NNEATTR2 = 0
OTHER_ATTR2 = 0


MOD_CLS = 0
EXT_CLS = 0
FUN_CLS = 0
FUN_CLS2 = 0

DEL_CLS = 0
VV1 = []
VV2 = []
VS1 = defaultdict(int)
VS2 = defaultdict(int)
VS3 = defaultdict(int)
DIFF = [0, 0, 0]
DIFF2 = [0, 0, 0]
V1 = []
V2 = []
V3 = []
V4 = []
P = []
M1C = []
M2C = []
M3C = []
M4C = []
M5C = []
M6C = []
M7C = []
O1C = []
O2C = []
O3C = []
O4C = []
O5C = []
O6C = []
O7C = []
SAM = []
SAO = []
I0 = []
I1 = []
I2 = []
I3 = []
DEPTH_ATTR = []
DEPTH_ATTR_REPO = defaultdict(list)
DEPTH_ATTR_LEN = defaultdict(int)

DEPTH_DESCR = []
DEPTH_DESCR_LEN = defaultdict(int)
SS = []
SC = []
DO = []
DS = [4,5,6]
EXP1 = []
IMP_MEM1 = []
IMP_FUN1 = []
IMP_ALI1 = []
EXP2 = []
IMP_MEM2 = []
IMP_FUN2 = []
IMP_ALI2 = []

EXT_SET = []
EXT_EVENT = []
GET_SET = defaultdict(int)
METHODS_DESCR = defaultdict(int)
CMETHODS_DESCR = defaultdict(int)
WRAPPER_DESCR = defaultdict(int)

GETSET = set()
GET_SET2 = defaultdict(int)
METHODS_DESCR2 = defaultdict(int)
CMETHODS_DESCR2 = defaultdict(int)
WRAPPER_DESCR2 = defaultdict(int)
NNUM = 0
SNUM = 0

DOP = {'insert':[], 'replacement': [], 'union': [], 'no_func': []}
FUN = {'setattr': [], 'loads': [], 'no_func': [] , 'hasattr': [], 'getattr': [], 'infer_dtype': [], 'insert': []}
#
CPATTERN = {'override':defaultdict(int), 'update':defaultdict(int), 'inv':defaultdict(int), 'mono':defaultdict(int)}
CPATTERN_DUP = {'override':defaultdict(set), 'update':defaultdict(set), 'inv':defaultdict(set), 'mono':defaultdict(set)}
APATTERN = {'extend':defaultdict(int), 'update':defaultdict(int), 'evolve': defaultdict(int),}
APATTERN_DUP = {'extend':defaultdict(set), 'update':defaultdict(set), 'evolve': defaultdict(set),}

class_hierarchy = None
class_attributes = None
class_attributes_no_objects = None
class_attributes_no_inheritance = None
OBJ_ATTRS = {'__new__', '__init__', '__ne__', '__class__', '__dict__', '__eq__', '__reduce__', '__sizeof__', '__dir__', '__format__', 
'__annotations__', '__str__', '__reduce_ex__', '__setattr__', '__module__', '__init_subclass__', '__repr__', '__delattr__', '__hash__', '__getattribute__', '__doc__'}

CA = {}
BUILTIN_DESCRIPTORS = ["builtins.member_descriptor", "builtins.property", "builtins.getset_descriptor", "builtins.function", "builtins.classmethod", 
"builtins.method_descriptor", "builtins.wrapper_descriptor", "builtins.classmethod_descriptor", "builtins.staticmethod", "functools.cached_property", 
"functools.functools._lru_cache_wrapper", "others"]
OTHERS_SET = set()
OTHERS_GET = set()

# SHOW_DESCR = {"builtins.member_descriptor": "member", "builtins.property": "property", "builtins.getset_descriptor", "getset", }
LDALL = {}
SDALL = {}
for descr in BUILTIN_DESCRIPTORS:
    SDALL[descr] = [0, 0, 0, 0]
    LDALL[descr] = [0, 0, 0, 0]
RNS = set()
LNS = set()
c = None
with open("file_funcs", "rb") as f:
    file_funcs = pkl.load(f)
# file_funcs = {}
with open("normal_variable_mutation", 'w+') as f:
    pass
with open("mutation_store_fast", 'w+') as f:
    pass

info_set = set()
def clear():
    global cnt, nob, ob, nn, info_set, mutative, cover, mutative_set, cover_set, ln, rn, modification, creation, extension, type_changing_mod, cnts
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
    cnts = defaultdict(int)

def print_info():
    # if cover > 0:
    #     print(f"mutative: {mutative}")
    #     print(f"portion: {mutative/cover}")
    #     print(f"portion set: {len(mutative_set)/len(cover_set)}")
    print(f"none none: {nn}")
    print(f"from none: {ln}")
    print(f"to none: {rn}")
    print(f"normal mutation: {ob}")

    print(f"modification: {modification}")
    
    print(f"creation: {creation}")
    print(f"extension: {extension}")
    
    pass
def add_object_back(ch):
    for k in ch:
        if 'builtins.object' not in ch[k]:
            ch[k].add('builtins.object')
# not actually c3 for now. 
def mro(x) -> set:
    global class_hierarchy
    if x == "builtins.object":
        return {x}
    if x not in class_hierarchy:
        return {x, "builtins.object"}
    r = {x}
    for s in class_hierarchy[x]:
        r.update(mro(s))
    return r

# a <n: b
def subtype(a, b):
    if a == b:
        return True
    mro_a = mro(a)
    return b in mro_a


def elca(a, b):
    # get the mro of a
    mro_a = mro(a)
    mro_b = mro(b)
    # get the mro of b
    common_superclasss = mro_a.intersection(mro_b)
    return common_superclasss

# a /\_n b
def nominal_join(a, b):
    return len(elca(a, b)) > 1

def nominal_join2(a, b, global_class_classes):
    if a in global_class_classes and b in global_class_classes:
        mro1 = {x[0] for x in global_class_classes[a].mro}
        mro2 = {x[0] for x in global_class_classes[b].mro}
        return len(mro1.intersection(mro2)) > 1
    else:
        return False

def generalized_nominal_join2(T, global_class_classes):
    if all(x in global_class_classes for x in T):
        mros = [{x[0] for x in global_class_classes[a].mro} for a in T]
        mro1 = mros[0]
        for mro2 in mros[1:]:
            mro1 = mro1.intersection(mro2)
        return len(mro1) > 1
    else:
        return False



def generalized_nominal_join2_(T, global_class_classes):
    if all(x in global_class_classes for x in T):
        mros = [{x[0] for x in global_class_classes[a].mro} for a in T]
        mro1 = mros[0]
        for mro2 in mros[1:]:
            mro1 = mro1.intersection(mro2)
        return mro1
    else:
        return False

def stru_join2(a, b, global_class_classes):
    if a in global_class_classes and b in global_class_classes:
        attr1 = set()
        for c in global_class_classes[a].mro:
            attr1.update(c[1].keys())
        attr2 = set()
        for c in global_class_classes[b].mro:
            attr2.update(c[1].keys())
        common_attrs = attr1.intersection(attr2)
        OBJ_ATTRS = set(global_class_classes[a].mro[-1][1].keys())
        return common_attrs.issuperset(OBJ_ATTRS) and len(common_attrs.difference(OBJ_ATTRS)) > 0
    else:
        return False

def generalized_stru_join2(T, global_class_classes):
    if all(x in global_class_classes for x in T):
        a = T[0]
        attr1 = set()
        for c in global_class_classes[a].mro:
            attr1.update(c[1].keys())
        for b in T[1:]:
            attr2 = set()
            for c in global_class_classes[b].mro:
                attr2.update(c[1].keys())
            attr1 = attr1.intersection(attr2)
        OBJ_ATTRS = set(global_class_classes[a].mro[-1][1].keys())
        return attr1.issuperset(OBJ_ATTRS) and len(attr1.difference(OBJ_ATTRS)) > 0
    else:
        return False

def generalized_stru_join2_(T, global_class_classes):
    if all(x in global_class_classes for x in T):
        a = T[0]
        attr1 = set()
        for c in global_class_classes[a].mro:
            attr1.update(c[1].keys())
        for b in T[1:]:
            attr2 = set()
            for c in global_class_classes[b].mro:
                attr2.update(c[1].keys())
            attr1 = attr1.intersection(attr2)
        OBJ_ATTRS = set(global_class_classes[a].mro[-1][1].keys())
        return attr1
    else:
        return False
def numberic_coercion(a, b):
    if a in ['builtins.float', 'builtins.int', 'numpy.numpy.float64', 'numpy.numpy.float32', 'numpy.numpy.int64', 'numpy.numpy.int32']:
        if b in ['builtins.float', 'builtins.int', 'numpy.numpy.float64', 'numpy.numpy.float32', 'numpy.numpy.int64', 'numpy.numpy.int32']:
            return True
    return False
def generalized_numberic_coercion(T):
    if all(x in ['builtins.float', 'builtins.int', 'numpy.numpy.float64', 'numpy.numpy.float32', 'numpy.numpy.int64', 'numpy.numpy.int32'] for x in T):
        return True
    return False
def nominal_supertype_static(a, b):
    return subtype(b, a)
def nominal_join_static(a, b):
    if nominal_supertype_static(a, b):
        return a
    elif nominal_supertype_static(b, a):
        return b
    else:
        mro1 = elca(a, b)
        assert len(mro1) > 0
        mro1 = list(mro1)
        best = mro1[0]
        for c in mro1:
            if len(class_hierarchy[c]) > len(class_hierarchy[best]):
                best = c
        return best
def generalized_nominal_join_static(T):
    assert len(T) >= 1
    base = T[0]
    for t in T[1:]:
        base = nominal_join_static(base, t)
    return base

# a /\_s b
def stru_join(nt1, nt2, t1, t2):
    # obj_attrs1 = obj_attr(t1)
    # obj_attrs2 = obj_attr(t2)
    cls_attrs1 = cls_attr(nt1)
    cls_attrs2 = cls_attr(nt2)
    
    attrs1 = cls_attrs1
    attrs2 = cls_attrs2

    common_attrs = attrs1.intersection(attrs2)
    return common_attrs.issuperset(OBJ_ATTRS) and len(common_attrs.difference(OBJ_ATTRS)) > 0


def invalid(T):
    for t in T:
        attr_map = repr_to_obj(t).children[2]
        t = fix_prefix(t)
        if not attr_map and t not in class_attributes:
            return True
    return False
def generalized_stru_join(T):
    T = copy(T)
    if 'builtins.NoneType' in T:
        T.remove('builtins.NoneType')
    if len(T) == 0 or invalid(T):
        return None
    obj_attrs = [obj_attr(t) for t in T]
    cls_attrs = [cls_attr(extract_nominal(t)) for t in T]
    attrs = [x.union(y) for x, y in zip(obj_attrs, cls_attrs)]
    
    attr = attrs[0]
    for attr2 in attrs[1:]:
        attr = attr.intersection(attr2)
    return attr.issuperset(OBJ_ATTRS) and len(attr.difference(OBJ_ATTRS)) > 0
def generalized_nomi_join(T):
    mros = [mro(t) for t in T]
    # get the mro of b
    common = mros[0]
    for m in mros[1:]:
        common.intersection_update(m)
    return len(common) > 1

def obj_attr(t):
    
    attr_map = repr_to_obj(t).children[2]
    if attr_map:
        return set(str(x.children[0]) for x in attr_map.children)
    else:
        return set()

def fix_prefix(t):
    
    t = t.replace('_io._io.', 'io.')
    t = t.replace('re.re.', 're.')
    t = t.replace('datetime.datetime.', 'datetime.')
    t = t.replace('numpy.numpy.', 'numpy.')
    
    return t
def cls_attr(t):
    t = fix_prefix(t)
    if t in class_attributes:
        return class_attributes[t]
    else:
        return set()

def depth_subtype(t1, t2):
    obj_attrs1 = obj_attr(t1)
    obj_attrs2 = obj_attr(t2)
    return True

def dot(amap, attr):
    for a in amap.children:
        if str(a.children[0]) == attr:
            return a.children[1]
def stru_subtype_obj(obj1, obj2):
    obj_attr_map1 = obj1.children[2]
    obj_attr_map2 = obj2.children[2]
    cls_attrs1 = cls_attr(str(obj1.children[0]))
    cls_attrs2 = cls_attr(str(obj2.children[0]))
    obj_attrs1 = set(str(x.children[0]) for x in obj_attr_map1.children) if obj_attr_map1 else set()
    obj_attrs2 = set(str(x.children[0]) for x in obj_attr_map2.children) if obj_attr_map2 else set()
    if cls_attrs1.issuperset(cls_attrs2) and obj_attrs1.issuperset(obj_attrs2):
        for obj_attr in obj_attrs2:
            if not stru_subtype_obj(dot(obj_attr_map1, obj_attr), dot(obj_attr_map2, obj_attr)):
                return False
        return True
    return False
# a <s: b
def stru_subtype(nt1, nt2, t1, t2):
    obj_attrs1 = obj_attr(t1)
    obj_attrs2 = obj_attr(t2)
    cls_attrs1 = cls_attr(nt1)
    cls_attrs2 = cls_attr(nt2)
    
    attrs1 = obj_attrs1.union(cls_attrs1)
    attrs2 = obj_attrs2.union(cls_attrs2)
    if len(attrs1) > 0 and len(attrs2) > 0:

        return attrs1.issuperset(attrs2) and depth_subtype(repr_to_obj(t1), repr_to_obj(t2))
    else:
        return False



def structural_coercion(a, b):
    if a in ['numpy.numpy.ndarray', 'builtins.list', 'builtins.dict', 'builtins.tuple', 'builtins.set']:
        if b in ['numpy.numpy.ndarray', 'builtins.list''builtins.dict', 'builtins.tuple', 'builtins.set']:
            return True
    
    if a in ['builtins.dict', 'requests.structures.CaseInsensitiveDict', 'collections.collections.OrderedDict', 'collections.collections.defaultdict']:
        if b in ['builtins.dict', 'requests.structures.CaseInsensitiveDict', 'collections.collections.OrderedDict', 'collections.collections.defaultdict']:
            return True

    return False
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
cache = {}

def exception_parse(repr):
    if repr == 'builtins.method-wrapper':
        repr = 'builtins.method_wrapper'
    if len(repr) > 7900:
        # fall back to nominal
        res = object_parser.parse(repr.split('<*>')[0].strip())
    else:
        try:
            res = object_parser.parse(repr)
        except UnexpectedCharacters as e:
            res = object_parser.parse(repr.split('<*>')[0].strip())
    return res
def extract_nominal(repr):
    if repr == 'builtins.method-wrapper':
        repr = 'builtins.method_wrapper'
    repr = repr.strip()
    global cache
    if repr in cache:
        return cache[repr].children[0]
    res = exception_parse(repr)
    cache[repr] = res
    return res.children[0]

def repr_to_obj(repr):
    repr = repr.strip()
    global cache
    if repr in cache:
        return cache[repr]
    res = exception_parse(repr)
    cache[repr] = res
    return res
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
    global c
    if c != 'store_attr':
        return frame_name
    return '<obj>:' + frame_name.split(":")[1]
max_lines = 0



def read_lines(path):
    
    lines = []
    with open(path, 'r', errors='ignore') as f:
        for line in tqdm(f):
            lines.append(line)
            if max_lines > 0 and len(lines) > max_lines:
                break
    return lines


def fix_master(master):
    global id_to_master
    if master.find('unknown_module.') != -1 or master.find('abc.') != -1 and master.split('-')[1] in id_to_master:
        return id_to_master[master.split('-')[1]]
    else:
        id_to_master[master.split('-')[1]] = master
        return master

        
def read_frames_biend(path):
    frames = []
    ok = defaultdict(int)
    cur = defaultdict(int)
    opcodes = defaultdict(list)
    infos = defaultdict(list)
    objects = defaultdict(list)
    object_dicts = defaultdict(list)
    serving = {}
    served = {}
    lines = read_lines(path)
    n = len(lines)
    ok['<obj>:<(nil)>'] = 1
    # for i in tqdm(range(n)):
    #     line = lines[i]

    #     if line.startswith("<opcode>"):
    #         linex = line.split("<opcode>")[1].strip()
    #         frame_name = ":".join(linex.strip().split(":")[:2])
    #         ok[frame_name] = 1
    global_id = 0
    for i in tqdm(range(n)):
        line = lines[i]
        if line.startswith("----------!"):
            frame_name = ":".join(line.split("----------!")[1].strip().split(":")[:2])
            frame_namex = change_obj(frame_name)
            if c == 'store_attr':
                short_master = frame_name.split(":")[0].split('-')[1]
            else:
                short_master = frame_name.split(":")[0]
            master = frame_name.split(":")[0] + ':' + str(object_reuses[short_master])
            master = fix_master(master)
            servant = frame_namex
            # serving[frame_namex] = frame_name.split(":")[0] + ':' + str(object_reuses[frame_name.split(":")[0]])
            serving[servant] = master
            served[short_master] = servant
            if c == 'store_attr':
                dict_services[servant].append(master)
            # opcodes[frame_namex].append('new dict')
            infos[master].append((global_id, 'init dict'))
            global_id += 1
        elif line.startswith("----------$"):
            frame_name = ":".join(line.split("----------$")[1].strip().split(":")[:2])
            frame_namex = change_obj(frame_name)
            if c == 'store_attr':
                short_master = frame_name.split(":")[0].split('-')[1]
            else:
                short_master = frame_name.split(":")[0]
            master = frame_name.split(":")[0] + ':' + str(object_reuses[short_master])
            master = fix_master(master)
            servant = frame_namex
            # serving[frame_namex] = frame_name.split(":")[0] + ':' + str(object_reuses[frame_name.split(":")[0]])
            serving[servant] = master
            served[short_master] = servant
            if c == 'store_attr':
                dict_services[servant].append(master)
            # opcodes[frame_namex].append('new dict')
            infos[master].append((global_id, 'assign dict'))
            global_id += 1
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
                object_reuses[short_master] += 1
                if short_master in served and served[short_master] != None:
                    serving[served[short_master]] = None
                    served[short_master] = None
                
            elif c == 'dict':
                if ok[frame_namex]:
                    serving[frame_namex] = None
        elif line.startswith("<opcode>"):
            linex = line.split("<opcode>")[1].strip()
            frame_name = ":".join(linex.strip().split(":")[:2])
            if frame_name in serving and serving[frame_name] is not None:
                opcodes[frame_name].append(line)
            # if c == 'store_attr':
            #     if frame_name in opcodes and frame_name in serving and serving[frame_name] is not None:
            #         opcodes[frame_name][-1].append(line)
            # else:
            #     if frame_name in opcodes and len(opcodes[frame_name]) > 0:
            #         opcodes[frame_name][-1].append(line)
        else:
            msg = line.split(':')
            # dict index
            if line[:5] == '<obj>':
                frame_name = ":".join(line.strip().split(":")[:2])
                if frame_name in serving and serving[frame_name] is not None:
                    infos[serving[frame_name]].append((global_id, line.strip()))
                    global_id += 1
            else:
                frame_name = line.strip().split(":")[0]
                if c == 'store_attr':
                    short_master = frame_name.split(":")[0].split('-')[1]
                else:
                    short_master = frame_name.split(":")[0]
                master = frame_name + ':' + str(object_reuses[short_master])
                master = fix_master(master)
                infos[master].append((global_id, line.strip()))
                global_id += 1
                # object index
            # if c == 'store_attr':
            #     if frame_name in infos and frame_name in serving and serving[frame_name] is not None:
            #         infos[frame_name][-1].append(line.strip())
            # else:
            #     if frame_name in opcodes and len(opcodes[frame_name]) > 0:
            #         infos[frame_name][-1].append(line.strip())
    # for frame_name in tqdm(opcodes):
    #     # if c == 'store_attr':
    #     #     assert len(opcodes[frame_name]) == len(dict_services[frame_name])
    #     if len(opcodes[frame_name]) == len(infos[frame_name]):
    #         n = len(opcodes[frame_name])
    #         seen = set()
    #         for i in range(n):
    #             # if len(opcodes[frame_name][i]) == len(infos[frame_name][i]):
    #             m = len(opcodes[frame_name][i])
    #             frame = tuple()
    #             for j in range(m):
    #                 idx = opcodes[frame_name][i][j]
    #                 info = infos[frame_name][i][j]
    #                 if True or ((idx.find(repo) != -1 or idx.find("<string>: __init__") != -1 ) and info.find(info_excludes[0]) == -1):
    #                     # good frame node
    #                     frame += ((idx, info),)
    #                     check_info_while_reading(idx, info)
    #             # if frame not in seen:
    #             #     # good frame
    #             #     seen.add(frame)
    #             frames.append(frame)
    #             if c == 'store_attr':
    #                 object_dicts[dict_services[frame_name][i]].append(frame)
    if c == 'store_attr':
        frames = infos
    return frames

usage_cache = {}
wp_cache = {}
wp_cache_only_cond = {}
assert_cache = {}
branch_cache = {}
func_node_cache = {}
def get_assert(f, func, l, attr):
    if (f, func, l, attr) not in assert_cache:
        analyzer = Analyzer(f)
        usage = analyzer.check_assert(func, int(l), attr)
        assert_cache[(f, func, l, attr)] = usage
    return assert_cache[(f, func, l, attr)]
def get_usage(f, func, l, attr):
    if (f, func, l, attr) not in usage_cache:
        analyzer = Analyzer(f)
        usage = analyzer.check_non_trival(func, int(l), attr)
        usage_cache[(f, func, l, attr)] = usage
    return usage_cache[(f, func, l, attr)]

def branch_check(f, func, l, attr):
    if (f, func, l, attr) not in branch_cache:
        analyzer = Analyzer(f)
        all_branch = analyzer.check_branch(func, int(l), attr)
        branch_cache[(f, func, l, attr)] = all_branch
    return branch_cache[(f, func, l, attr)]
def get_wp(f, func, l, attr):
    if (f, func, l, attr) not in wp_cache:
        analyzer = Analyzer(f)
        wp, name = analyzer.get_wp_for_loc(func, int(l), attr)
        wp_cache[(f, func, l, attr)] = (wp, name)
    return wp_cache[(f, func, l, attr)]

def get_wp_only_cond(f, func, l, attr):
    if (f, func, l, attr) not in wp_cache_only_cond:
        analyzer = Analyzer(f)
        wp, name = analyzer.get_wp_for_loc(func, int(l), attr, consider_raise = False)
        wp_cache_only_cond[(f, func, l, attr)] = (wp, name)
    return wp_cache_only_cond[(f, func, l, attr)]

def get_func_node(f, func, l):
    if (f, func, l) not in func_node_cache:
        analyzer = Analyzer(f)
        func_node = analyzer.get_func_node(func, int(l))
        func_node_cache[(f, func, l)] = func_node
    return func_node_cache[(f, func, l)]
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
    new_idx[frame_id] = new_idx[frame_id][:-15]
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
    global nn, ln, rn, nob, ob, mutative, cover, modification, creation, extension, type_changing_mod, class_objects, obj, ext, extensive_class, attr_type_change, deleting_class, clearing_class
    global t1st2, t2st1, t1et2, modfying_class, t1st2s, t2st1s, obs, t1et2s, nobs, lns, rns, all_muts
    global obs
    global vt, extensive, deleting, clearing, modfying, type_env, dangerous_ref
    global i0, i1, i2, i3
    global self_ext_set, ext_set
    global refcnts, maxrefcnts
    refcnt = 1
    maxrefcnt = 1
    for info in frame:
        if info == 'new dict':
            type_env.clear()
        else:
            cover += 1
            msg = info.split(":")
            if len(msg) == 7:
                
                name, attr, t = msg[1].strip(), msg[4].strip(), msg[5].strip()

                if msg[6].strip() == 'update':
                    a = 1
                if msg[6].strip() == 'insert 0':
                    i0 += 1
                if msg[6].strip() == 'insert 1':
                    i1 += 1
                if msg[6].strip() == 'insert 2':
                    i2 += 1
                if msg[6].strip() == 'insert 3':
                    i3 += 1
                
                if attr in type_env:
                    if t != type_env[attr]:
                        # modification
                        if refcnt > 4:
                            dangerous_ref = 1 
                        modfying = 1
                        modification += 1
                        nt1 = extract_nominal(type_env[attr])
                        nt2 = extract_nominal(t)
                        # nt1 = type_env[attr]
                        # nt2 = t
                        mutative += 1
                        attr_type_change[(type_env[attr], t)] += 1 
                        all_muts.add((type_env[attr], t))
                        if type_env[attr] != 'builtins.NoneType' and t != 'builtins.NoneType':
                            if nt1 == nt2:
                                t1et2 += 1 
                                t1et2s.add((type_env[attr], t))

                            elif subtype(nt1, nt2):
                                t1st2 += 1
                                t1st2s.add((type_env[attr], t))
                            elif subtype(nt2, nt1):
                                t2st1 += 1
                                t2st1s.add((type_env[attr], t))
                            else:
                                common_superclasss = elca(nt1, nt2)
                                if len(common_superclasss) > 1:
                                    nob += 1
                                    nobs.add((type_env[attr], t))
                                else:
                                    with open(f'others_{repo}.txt', "a+") as f:
                                        f.write(nt1+ '->' + nt2 + '\n')
                                    ob += 1
                                    obs.add((type_env[attr], t))
                        else:
                            if type_env[attr] == 'builtins.NoneType':
                                ln += 1
                                lns.add((type_env[attr], t))
                            else:
                                rn += 1
                                rns.add((type_env[attr], t))
                else:
                    func = msg[3].strip()[:12]
                    # if func == 'self-initing' or func.find('copy') != -1 or idx.find('new') != -1: 
                    if func == 'self-initing': 
                        creation += 1
                    else:
                        extension += 1
                        extensive = 1
                        if refcnt > 4:
                            dangerous_ref = 1 
                type_env[attr] = t
                if attr in vt:    
                    vt[attr].add(t)
                else:
                    vt[attr] = {t}
            elif len(msg) == 6:
                # assert False
                print(info)
                attr = msg[4].strip()
                if attr in type_env:
                    del type_env[attr]
                deleting = 1
                if refcnt > 4:
                    dangerous_ref = 1 
            elif len(msg) == 5:
                type_env.clear()
                clearing = 1
                if refcnt > 4:
                    dangerous_ref = 1 
            elif len(msg) == 3:
                if msg[2].strip().find('increase') != -1:
                    refcnt += 1
                elif msg[2].find('decrease') != -1:
                    refcnt -= 1
                maxrefcnt = max(maxrefcnt, refcnt)

    refcnts[refcnt] += 1
    maxrefcnts[maxrefcnt] += 1

def only_type(keys):
    return {k.split("-")[1] for k in keys}


def _analyze_dict(frame):
    global nn, ln, rn, nob, ob, mutative, cover, modification, creation, extension, type_changing_mod, only_create, not_only_create, hete_value, hete_key, delete_dict, clear_dict
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
CPLOYMORPHISM = []
APLOYMORPHISM = defaultdict(int)
APLOYMORPHISM_POT = defaultdict(int)

OAPLOYMORPHISM = defaultdict(int)
OAPLOYMORPHISM_POT = defaultdict(int)

obj = None
ext = 0
del_ = 0
mod = 0
clr = 0
t1st2 = 0
i0 = 0
i1 = 0
i2 = 0
i3 = 0
t2st1 = 0
t1et2 = 0
extensive_class = set()
deleting_class = set()
clearing_class = set()
replacing_class = set()
modfying_class = set()
vt = {}
extensive = 0
deleting = 0
clearing = 0
modfying = 0
dangerous_ref = 0
type_env = {}
attr_type_change = defaultdict(int)
obs = set()
t1st2s = set()
t2st1s = set()
t1et2s = set()
nobs = set()
lns = set()
rns = set()
all_muts = set()
self_ext_set = set()
ext_set = set()
refcnts = defaultdict(int)
maxrefcnts = defaultdict(int)

def floor_to(x):
    return floor(x*1000) / 1000

def ceil_to(x):
    return ceil(x*10000) / 10000

def sci_not(x):
    if x == -1:
        return '*'
    if USE_SCI:
        x = "{:e}" .format (x)
        return x[0:4] + x[8:]
    if USE_PERC:
        return str(floor(x * 1000)/10)
    return "%.3f" % x

def div(a, b):
    if b == 0:
        return 0
    else:
        return a / b


def attr_load_events():
    ale = defaultdict(set)
    # with open(f"/home/user/purepython/cpython-3.9/pydyna/store_attr_load_{repo}.txt", errors = 'ignore') as f:
    #     for line in f:
    #         msg = line.strip().split(":")
    #         ale[msg[0].strip()].add(msg[4].split('.')[-1])
    return ale

def extract_mro(mro_str):
    # idx = mro_str.find("[")
    # if idx != -1:
    #     mro_str = mro_str[:idx]
    mro = []
    t = mro_parser.parse(mro_str)
    for c in t.children:
        mro.append((str(c.children[0]), c.children[2]))
    return mro
def load_mro():
    mro = {}
    t = time.time()
    with open(f"/home/user/purepython/cpython-3.9/pydyna/store_attr_mro_{repo}.txt") as f:
        for line in f:
            try:
                msg = line.strip().split(":")
                if msg[0].strip() not in mro:
                    mro[msg[0].strip()] = extract_mro(msg[1].strip())
            except Exception as e:
                pass
    eprint(f"load mro cost: {time.time()-t}")
    return mro

def load_cls_elv():
    cls_elv = defaultdict(list)
    t = time.time()
    with open(f"/home/user/purepython/cpython-3.9/pydyna/store_attr_flow_err_{repo}.txt") as f:
        for line in f:
            msg = line.strip().split(":")
            if len(msg) == 6:
                cls_elv[msg[1].strip()].append(msg)
    eprint(f"load cls elv cost: {time.time()-t}")
    return cls_elv


def missable(class_objects, cls, attr):
    cls = fix_prefix(cls)
    if cls not in class_attributes:
        return False
    return any(attr in dic for dic in class_objects[cls]) and not all(attr in dic for dic in class_objects[cls]) and attr not in class_attributes[cls]

def read_error_object():
    r = defaultdict(set)
    with open(f'error_objects/{repo}') as f:
        for line in f:
            clas, num = line.split()
            r[clas].add(int(num))
    return r


def exact_term(name, attr, expr):
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name):
            if expr.value.id == name and expr.attr == attr:
                return True
    return False

def exact_name(name, attr, expr):
    if isinstance(expr, ast.Name):
        if expr.id == name:
            return True
    return False

def filter_type_not(all_type, name, attr, filt):
    global exact, comp_succ, comp_fail, call, unary, functioning_literal
    if exact_term(name, attr, filt):
        functioning_literal += 1
        exact += 1
        # with open('functioning_literal', 'a+') as f:
        #     f.write(pprint2.pprint_top(filt) + '\n')
        return {x for x in all_type if x != 'builtins.NoneType'}

    # with open('failed_term', 'a+') as f:
    #     f.write(name + '.' + attr + '   ' + pprint2.pprint_top(filt) + '\n')
    
    if isinstance(filt, ast.Compare):
        if len(filt.comparators) == 1:
            if exact_term(name, attr, filt.left) or exact_term(name, attr, filt.comparators[0]):
                # TODO
                if isinstance(filt.ops[0], ast.Is) or isinstance(filt.ops[0], ast.Eq):
                    if none_term(filt.left) or none_term(filt.comparators[0]):
                        new_type = {x for x in all_type if x != 'builtins.NoneType' and x != 'missable'}
                        comp_succ += 1
                        functioning_literal += 1
                        # with open('functioning_literal', 'a+') as f:
                        #     f.write(pprint2.pprint_top(filt) + '\n')
                        return new_type
                # print('???')
        # if pprint2.pprint_top(filt).find(attr) != -1:
        #     comp_fail += 1
    if isinstance(filt, ast.Call):
        if any(exact_term(name, attr, x) for x in filt.args):
            # TODO
            functioning_literal += 1
            # with open('functioning_literal', 'a+') as f:
            #     f.write(pprint2.pprint_top(filt) + '\n')
            call += 1
            return filter_type_call(all_type, name, attr, filt, isnot = True)
        if any(exact_name(name, attr, x) for x in filt.args):
            # TODO
            functioning_literal += 1
            # with open('functioning_literal', 'a+') as f:
            #     f.write(pprint2.pprint_top(filt) + '\n')
            call += 1
            return filter_type_call_name(all_type, name, attr, filt, isnot = True)
        
    return all_type
def none_term(expr):
    if isinstance(expr, ast.Constant) and expr.value == None:
        return True
    return False


def match_type(class_frag, all_type):
    all_type = [str(extract_nominal(x)) for x in all_type]
    # fuzzing match
    single_match = sum(t.find(class_frag) != -1 for t in all_type)
    if single_match >= 1:
        for t in all_type:
            if t.find(class_frag) != -1:
                return t
    if single_match > 1:
        assert False
    # exact match
    for name in class_hierarchy:
        if name.find(class_frag) != -1:
            return name
    
    return None

def get_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr

def subtype_ast(t1, node, all_type):

    if isinstance(node, ast.Name) or isinstance(node, ast.Attribute):
        types = [get_name(node)]
    else:
        if isinstance(node, (ast.Tuple, ast.List)) and all(isinstance(x, (ast.Name, ast.Attribute)) for x in node.elts):
            types = [get_name(x) for x in node.elts]
        else:
            raise TypeError()
    for t in types:
        expected_type = match_type(t, all_type)
        if subtype(str(extract_nominal(t1)), expected_type):
            return True
    return False
def filter_type_call(all_type, name, attr, filt, isnot = False):
    if isinstance(filt.func, ast.Name):
        if filt.func.id == "isinstance":
            try:
                if isnot:
                    r = {x for x in all_type if subtype_ast(x, filt.args[1], all_type)}
                else:
                    r = {x for x in all_type if not subtype_ast(x, filt.args[1], all_type)}
            except TypeError as e:
                r = all_type
            return r
    return all_type

def filter_type_call_name(all_type, name, attr, filt, isnot = False):
    if isinstance(filt.func, ast.Name):
        if filt.func.id == "hasattr":
            try:
                if isnot:
                    r = {x for x in all_type if x != 'missing'}
                else:
                    r = {x for x in all_type if x == 'missing'}
            except TypeError as e:
                r = all_type
            return r
    return all_type
exact = comp_succ = comp_fail = call = unary = related_literal = functioning_literal = 0
def filter_type_literal(all_type, name, attr, filt):
    global exact, comp_succ, comp_fail, call, unary, related_literal, functioning_literal
    # types that satisify filt should be removed
    # if pprint2.pprint_top(filt).find(attr) != -1:
    #     related_literal += 1
    #     with open('related_literal', 'a+') as f:
    #         f.write(pprint2.pprint_top(filt)+'\n')
    if isinstance(filt, ast.UnaryOp):
        unary += 1
        if isinstance(filt.op, ast.Not):
            return filter_type_not(all_type, name, attr, filt.operand)
        else:
            return all_type
    else:
        if exact_term(name, attr, filt):
            # all_type.remove('builtins.NoneType')
            # exact += 1
            return all_type
        if isinstance(filt, ast.Compare):
            if len(filt.comparators) == 1:
                if exact_term(name, attr, filt.left) or exact_term(name, attr, filt.comparators[0]):
                    # TODO
                    if isinstance(filt.ops[0], ast.Is) or isinstance(filt.ops[0], ast.Eq):
                        if none_term(filt.left) or none_term(filt.comparators[0]):
                            comp_succ += 1
                            functioning_literal += 1
                            # with open('functioning_literal', 'a+') as f:
                            #     f.write(pprint2.pprint_top(filt)+'\n')
                            return {x for x in all_type if x != 'builtins.NoneType'}
                    # print('???')
            # if pprint2.pprint_top(filt).find(attr) != -1:
            #     comp_fail += 1
        if isinstance(filt, ast.Call):
            if any(exact_term(name, attr, x) for x in filt.args):
                # TODO
                # print('???')
                functioning_literal += 1
                # with open('functioning_literal', 'a+') as f:
                #     f.write(pprint2.pprint_top(filt)+'\n')
                call += 1
                return filter_type_call(all_type, name, attr, filt)
            

        return all_type

clauses = 0
singular_clauses = 0
def filter_type_clause(all_type, name, attr, clause):

    if isinstance(clause, ast.BoolOp) and isinstance(clause.op, ast.Or):
        all_types = filter_type_clause(all_type, name, attr, clause.values[0])
        for lit in clause.values[1:]:
            all_types.intersection_update(filter_type_clause(all_type, name, attr, lit))
        return all_types
    else:
        if isinstance(clause, (ast.UnaryOp, ast.Name, ast.Compare, ast.Call, ast.Attribute, ast.Subscript)):
            return filter_type_literal(all_type, name, attr, clause)
        else:
            return all_type
    global clauses, singular_clauses
    filt = None
    clauses += 1
    if isinstance(clause, ast.BoolOp) and len(clause.values) == 1 and isinstance(clause.values[0], (ast.Name, ast.Compare, ast.Call, ast.Attribute)):
        filt = clause.values[0]
    elif isinstance(clause, (ast.Name, ast.Compare, ast.Call, ast.Attribute, ast.UnaryOp)):
        filt = clause
    if filt:
        singular_clauses += 1
        return filter_type_literal(all_type, name, attr, filt)
    else:
        return all_type

def filter_type(all_type, name, attr, cnf_wp):
    for clause in cnf_wp.values:
        all_type = filter_type_clause(all_type, name, attr, clause)
    return all_type
# class HasattrFinder(ast.NodeVisitor):
#     def __init__(self, name, attr, node) -> None:
#         super().__init__()
#         self.attr = attr
#         self.name = name
#         self.ok = False
#         self.visit(node)

#     def visit_Call(self, node):
#         if isinstance(node.func, ast.Name) and node.func.id == 'hasattr': 
#             if any(exact_name(self.name, self.attr, x) for x in node.args):
#                 self.ok = True
builtin_names = ['hasattr', 'isinstance']
class HasattrFinder(ast.NodeVisitor):
    def __init__(self, name, attr, node) -> None:
        super().__init__()
        self.attr = attr
        self.name = name
        self.ok = True
        self.visit(node)

    def visit_Name(self, node:ast.Name):
        if node.id != self.name and node.id not in builtin_names: 
            self.ok = False
def checked_condition_literal(name, attr, filt):
    if isinstance(filt, ast.UnaryOp):
        if isinstance(filt.op, ast.Not):
            filt = filt.operand
        else:
            return False
    if exact_term(name, attr, filt):
        return True
    elif isinstance(filt, ast.Compare):
        if len(filt.comparators) == 1:
            if exact_term(name, attr, filt.left) or exact_term(name, attr, filt.comparators[0]):
                if isinstance(filt.ops[0], ast.Is) or isinstance(filt.ops[0], ast.Eq) or isinstance(filt.ops[0], ast.IsNot):
                    if none_term(filt.left) or none_term(filt.comparators[0]):
                        return True
                
        return False
    elif isinstance(filt, ast.Call):
        if any(exact_term(name, attr, x) for x in filt.args):
            if isinstance(filt.func, ast.Name) and filt.func.id == 'hasattr':
                return True
    return False
def checked_condition_clause(name, attr, clause):

    if isinstance(clause, ast.BoolOp) and isinstance(clause.op, ast.Or):
        check_others = checked_condition_clause(name, attr, clause.values[0])
        for lit in clause.values[1:]:
            check_others &= checked_condition_clause(name, attr, lit)
        return check_others
    else:
        if isinstance(clause, (ast.UnaryOp, ast.Name, ast.Compare, ast.Call, ast.Attribute, ast.Subscript)):
            return checked_condition_literal(name, attr, clause)
        else:
            return False
def checked_condition(name, attr, cnf_wp):
    find_ = True
    for clause in cnf_wp.values:
        # ok = HasattrFinder(name, attr, clause).ok
        ok = checked_condition_clause(name, attr, clause)
        # ok = name in pprint2.pprint_top(clause) and attr in pprint2.pprint_top(clause) 
        find_ &= ok
    return find_

# def checked_condition(name, attr, wp):
    
#     find_ = False
#     for clause in cnf_wp.values:
#         # ok = HasattrFinder(name, attr, clause).ok
#         ok = name in pprint2.pprint_top(clause) and attr in pprint2.pprint_top(clause) 
#         find_ |= ok
#     return find_


def str_union(T):
    return '\t'.join(T)

def alias_pattern(pattern, loc, imp_mem1, imp_mem1_s, imp_func1, imp_func1_s, imp_ali1, imp_ali1_s, exp1, exp1_s, refcnt, local_, mode):
    member_method, para = member_para(pattern)
    if para: 
        # para
        if member_method:
            imp_mem1[0] += 1
            imp_mem1_s.add(loc)
            APATTERN[mode]['method'] += 1
            APATTERN_DUP[mode]['method'].add(loc)
            
        else:
            imp_func1[0] += 1
            imp_func1_s.add(loc)
            APATTERN[mode]['function'] += 1
            APATTERN_DUP[mode]['function'].add(loc)
    else:

        if local_ == 0:
            imp_ali1[0] += 1
            imp_ali1_s.add(loc)
            APATTERN[mode]['non_local'] += 1
            APATTERN_DUP[mode]['non_local'].add(loc)
        else:
            exp1[0] += 1
            exp1_s.add(loc)
            APATTERN[mode]['local'] += 1
            APATTERN_DUP[mode]['local'].add(loc)
        

# def numberic_coercion(nt1, nt2):
#     if nt1 in ['builtins.int', 'builtins.float'] and nt2 in ['builtins.int', 'builtins.float']:
#         return True
def not_same_class(raw1, raw2):
    t1 = raw1.strip()
    t = raw2.strip()
    nt1 = str(extract_nominal(raw1))
    nt2 = str(extract_nominal(raw2))
    return nt1 != nt2 # and not numberic_coercion(nt1, nt2)
def type_change_pattern(raw1, raw2, t1et2, t1et2s, t1st2, t1st2s, t2st1, t2st1s, nob, nobs, ob, obs, ln, lns, rn, rns, all_mut, all_muts, stru, strus, mode, loc, global_class_classes):
    t1 = raw1.strip()
    t = raw2.strip()
    nt1 = str(extract_nominal(t1))
    nt2 = str(extract_nominal(t))
    identify = loc
    all_mut[0] += 1
    all_muts.add(identify)
    sub = False
    mono = False
    if t1 != 'builtins.NoneType' and t != 'builtins.NoneType':


        if nt1 == nt2:
            t1et2[0] += 1 
            t1et2s.add(identify)

        elif subtype(nt1, nt2):
            t1st2[0] += 1
            t1st2s.add(identify)
            CPATTERN[mode]['t1st2'] += 1
            CPATTERN_DUP[mode]['t1st2'].add(loc)
        elif subtype(nt2, nt1):
            t2st1[0] += 1
            t2st1s.add(identify)
            CPATTERN[mode]['t2st1'] += 1
            CPATTERN_DUP[mode]['t2st1'].add(loc)

            sub = True
            mono = True
        elif nominal_join2(nt2, nt1, global_class_classes):
            nob[0] += 1
            nobs.add(identify)
            CPATTERN[mode]['t1simt2'] += 1
            CPATTERN_DUP[mode]['t1simt2'].add(loc)
        # elif stru_subtype_obj(repr_to_obj(t), repr_to_obj(t1)):
        #     # print(nt1)
        #     # print(nt2)
        #     stru[0] += 1
        #     strus.add(identify)
        #     CPATTERN[mode]['t1sst2'] += 1
        #     CPATTERN_DUP[mode]['t1sst2'].add(loc)
        #     # sub = True
        #     # mono = True
        # elif stru_subtype_obj(repr_to_obj(t1), repr_to_obj(t)):
        #     stru[0] += 1
        #     strus.add(identify)
        #     CPATTERN[mode]['t2sst1'] += 1
        #     CPATTERN_DUP[mode]['t2sst1'].add(loc)
        elif numberic_coercion(nt1, nt2):
            CPATTERN[mode]['coercion'] += 1
            CPATTERN_DUP[mode]['coercion'].add(loc)
            sub = True
            mono = True
        elif stru_join2(nt1, nt2, global_class_classes):
            nob[0] += 1
            nobs.add(identify)
            CPATTERN[mode]['t1stru2'] += 1
            CPATTERN_DUP[mode]['t1stru2'].add(loc)
        else:
            ob[0] += 1
            obs.add(identify)
            print(nt1)
            print(nt2)
            if mode == 'update':
                PAT[(nt1, nt2)].add(loc)
                PAT2[str(tuple([nt1, nt2]))] += 1
            CPATTERN[mode]['others'] += 1
            CPATTERN_DUP[mode]['others'].add(loc)
    else:
        if t1 == 'builtins.NoneType':
            ln[0] += 1
            lns.add(identify)
            LNS.add(identify)
            CPATTERN[mode]['ln'] += 1
            CPATTERN_DUP[mode]['ln'].add(loc)
            sub = True
        else:
            rn[0] += 1
            rns.add(identify)
            RNS.add(identify)
            CPATTERN[mode]['rn'] += 1
            CPATTERN_DUP[mode]['rn'].add(loc)

    return sub, mono
def class_analysis():
    global M1C, M2C, M3C, M4C, M5C, M6C, M7C
    # ssum = 0
    # all_cls = set()
    # for clas in class_attributes_no_objects:
    #     if clas.find(repo) != -1:
    #         all_cls.add(clas)
    #         if clas in class_attributes_no_objects and '__setattr__' in class_attributes_no_objects[clas]:
    #             ssum += 1
    #             DO.append(clas)
    #         else:
    #             ssum += 0
    

    # print(f"dot overloading: {ssum / len(all_cls)}")
    # # DO.append(ssum)

    # ssum = 0
    # for clas in class_attributes_no_objects:
    #     if clas.find(repo) != -1:
    #         if clas in class_attributes_no_inheritance and '__slots__' in class_attributes_no_inheritance[clas]:
    #             ssum += 1
    #         else:
    #             ssum += 0
    # print(f"static classes: {ssum / len(all_cls)}")
    # SC.append(ssum / len(all_cls))

    sames = []
    not_sames = []
    mutating = set()
    nob = ob = ln = rn = t1st2 = t2st1 = t1et2 = 0
    nobs, obs, lns, rns, t1st2s, t2st1s, t1et2s, all_muts = (set() for i in range(8))
    with open(f"/home/user/purepython/cpython-3.9/pydyna/store_attr_flow_err_{repo}.txt") as f:
        lines = f.readlines()
        for line in lines:
            msg = line.split(":")
            if len(msg) == 4:
                # overloading
                if msg[0].strip() != 'mutating':
                    pass
                else:
                    if msg[2].strip() != msg[3].strip():
                        not_sames.append(line)
                        t1 = msg[2].strip()
                        t = msg[3].strip()
                        nt1 = str(extract_nominal(t1))
                        nt2 = str(extract_nominal(t))
                        if t1 != 'builtins.NoneType' and t != 'builtins.NoneType':
                            if nt1 == nt2:
                                t1et2 += 1 
                                t1et2s.add((t1, t))

                            elif subtype(nt1, nt2):
                                t1st2 += 1
                                t1st2s.add((t1, t))
                            elif subtype(nt2, nt1):
                                t2st1 += 1
                                t2st1s.add((t1, t))
                            else:
                                common_superclasss = elca(nt1, nt2)
                                if len(common_superclasss) > 1:
                                    nob += 1
                                    nobs.add((t1, t))
                                else:
                                    ob += 1
                                    obs.add((t1, t))
                        else:
                            if t1 == 'builtins.NoneType':
                                ln += 1
                                lns.add((t1, t))
                            else:
                                rn += 1
                                rns.add((t1, t))
                    else:
                        sames.append(line)
    print(len(not_sames))
    print(len(mutating))
    mutative = len(not_sames)
    ss = mutative + len(sames)
    M1C.append(sci_not(round(div(ln, mutative), 3)))
    M2C.append(sci_not(round(div(rn, mutative), 3)))
    M3C.append(sci_not(round(div(t1et2 ,  mutative), 3)))
    M4C.append(sci_not(round(div(t1st2 ,  mutative), 3)))
    M5C.append(sci_not(round(div(t2st1 ,  mutative), 3)))
    M6C.append(sci_not(round(div(nob ,  mutative), 3)))
    M7C.append(sci_not(round(div(ob, mutative), 3)))
    SAM.append(str(ss) + '/' + sci_not(round(div(len(sames), mutative + len(sames)), 3)))
    sames = []
    not_sames = []
    mutating = set()
    nob = ob = ln = rn = t1st2 = t2st1 = t1et2 = 0
    nobs, obs, lns, rns, t1st2s, t2st1s, t1et2s, all_muts = (set() for i in range(8))
    with open(f"/home/user/purepython/cpython-3.9/pydyna/store_attr_flow_err_{repo}.txt") as f:
        lines = f.readlines()
        for line in lines:
            msg = line.split(":")
            if len(msg) == 4:
                # overloading
                if msg[0].strip() == 'mutating':
                    pass
                else:
                    if msg[2].strip() != msg[3].strip():
                        not_sames.append(line)
                        t1 = msg[2].strip()
                        t = msg[3].strip()
                        nt1 = str(extract_nominal(t1))
                        nt2 = str(extract_nominal(t))
                        if t1 != 'builtins.NoneType' and t != 'builtins.NoneType':
                            if nt1 == nt2:
                                t1et2 += 1 
                                t1et2s.add((t1, t))

                            elif subtype(nt1, nt2):
                                t1st2 += 1
                                t1st2s.add((t1, t))
                            elif subtype(nt2, nt1):
                                t2st1 += 1
                                t2st1s.add((t1, t))
                            else:
                                common_superclasss = elca(nt1, nt2)
                                if len(common_superclasss) > 1:
                                    nob += 1
                                    nobs.add((t1, t))
                                else:
                                    ob += 1
                                    obs.add((t1, t))
                        else:
                            if t1 == 'builtins.NoneType':
                                ln += 1
                                lns.add((t1, t))
                            else:
                                rn += 1
                                rns.add((t1, t))
                    else:
                        sames.append(line)
    print(len(not_sames))
    print(len(mutating))
    mutative = len(not_sames)
    ss = mutative + len(sames)
    O1C.append(sci_not(round(div(ln, mutative), 3)))
    O2C.append(sci_not(round(div(rn, mutative), 3)))
    O3C.append(sci_not(round(div(t1et2 ,  mutative), 3)))
    O4C.append(sci_not(round(div(t1st2 ,  mutative), 3)))
    O5C.append(sci_not(round(div(t2st1 ,  mutative), 3)))
    O6C.append(sci_not(round(div(nob ,  mutative), 3)))
    O7C.append(sci_not(round(div(ob, mutative), 3)))
    SAO.append(str(ss) + '/' + sci_not(round(div(mutative, mutative + len(sames)), 3)))

def member_para(text):
    func = text.strip()
    member_method, para = func.split('*')
    member_method = int(member_method)
    para = int(para)
    return member_method, para


def mro_class(global_class_classes, M, dic):
    mro = []
    for class_name, attr_map in dic:
        class_attr_map = {}
        for c in attr_map.children:
            a = str(c.children[0])
            cc = str(c.children[1].children[0])
            class_attr_map[a] = {cc}
        mro.append((class_name, class_attr_map))
    return mro

def mro_type_write(global_class_classes, M, dic):
    for class_name, attr_map in dic:
        class_attr_map = {}
        for c in attr_map.children:
            add_type_write(global_class_classes, M, c.children[1])

def add_type_self(global_class_classes, mro, clas, attr, t):
    if clas not in global_class_classes:
        return 
        # global_class_classes[clas] = ClassType()
        # global_class_classes[clas].mro = mro_class(global_class_classes, mro, mro[clas])
    if attr not in global_class_classes[clas].attribute_map:
        global_class_classes[clas].attribute_map[attr] = set()
    global_class_classes[clas].attribute_map[attr].add(t)

def add_type( clas, vt, attr, t):
    if t == 'missing':
        if attr in vt:    
            vt[attr][0] = 1
        else:
            vt[attr] = [1, set()]
    else:
        if attr in vt:    
            if SAVE_NOMINAL:
                vt[attr][1].add(str(extract_nominal(t)))
            
            else:
                vt[attr][1].add(t)
            
        else:
            if SAVE_NOMINAL:
                vt[attr] = [0, {str(extract_nominal(t))}]
            else:
                vt[attr] = [0, {t}]

def add_type_write(global_class_classes, mro, t):
    clas = str(t.children[0])
    attr_repr = t.children[2]
    if attr_repr:
        attr_map = {"___": "____"}
        for c in attr_repr.children:
            a = str(c.children[0])
            cc = str(c.children[1].children[0])
            add_type_write(global_class_classes, mro, c.children[1])
            attr_map[a] = cc
    else:
        attr_map = {"___": "____"}
    for a, t in attr_map.items():
        add_type_self(global_class_classes, mro, clas, a, t)
def write_obj(obj, f):
    for attr, typ in obj:
        f.write(attr + ' : ' + str(typ) + '\n')
    f.write('--------------------\n')


def none_or_another(objs):
    classes = {str(extract_nominal(obj)) for obj in objs}
    return len(classes) == 2 and 'builtins.NoneType' in classes
def multiple_class(objs):
    classes = {str(extract_nominal(obj)) for obj in objs}
    return len(classes) > 1 and not generalized_nomi_join(classes)
def attr_in_obj(obj, attr):
    oa = obj_attr(obj)
    ca = cls_attr(str(extract_nominal(obj)))
    return attr in oa or attr in ca or str(extract_nominal(obj)).find('test') != -1

def inclass(attr, mro):
    for c, am in mro:
        if attr in am:
            return True
    return False

def ok(objs):
    for obj in objs:
        if obj != 'missing':
            if obj not in global_class_classes:
                return False
    if len(objs) == 2 and 'missing' in objs and 'builtins.NoneType' in objs:
        return False
    return True
def attr_in_obj_nominal(obj, attr, nominal):
    if obj in global_class_classes:
        # do not consider missing here
        return attr in global_class_classes[obj].attribute_map or inclass(attr, global_class_classes[obj].mro)
    elif nominal:
        # fall back to static class
        return attr in class_attributes[obj]
    else:
        return False
def find_attr(objs, attr, nominal = False):
    if 'missing' in objs:
        return False
    if len(objs) == 1 and not nominal:
        # a shortcut 
        return True
    attr = list(attr)[0]
    return all(attr_in_obj_nominal(obj, attr, nominal) for obj in objs)

def filter_type_expr(all_type, name, attr, expr):
    attr_str = name + '.' + attr
    attr_is_none = attr_str + ' is None'
    attr_isnot_none = attr_str + ' is not None'
    if (attr_is_none in expr or attr_isnot_none in expr) and 'builtins.NoneType' in all_type:
        all_type.remove('builtins.NoneType')
    attr_is_none = attr_str + ' == None'
    attr_isnot_none = attr_str + ' != None'
    if (attr_is_none in expr or attr_isnot_none in expr) and 'builtins.NoneType' in all_type:
        all_type.remove('builtins.NoneType')
    return all_type

def remove_none(reachable_type):
    if 'builtins.NoneType' in reachable_type:
        reachable_type.remove('builtins.NoneType')
    return reachable_type
def remove_null(reachable_type):
    if 'missing' in reachable_type:
        reachable_type.remove('missing')
    return reachable_type
# def remove_none_and_missing(reachable_type):
#     if 'builtins.NoneType' in reachable_type:
#         reachable_type.remove('builtins.NoneType')
#     if 'missing' in reachable_type:
#         reachable_type.remove('missing')
#     return reachable_type


def class_func(events):
    for msg in events:
        if msg[-1].strip() == msg[-2].strip() and msg[-1].strip() in POSSIBLE_FUNC:
            return True
    return False

def class_extend(events):
    for msg in events:
        if msg[-1].strip() != msg[-2].strip() and  msg[-2].strip() == "missing":
            return True
    return False

def class_modify(events):
    for msg in events:
        if msg[-1].strip() != msg[-2].strip() and  msg[-1].strip() != "c2":
            return True
    return False

def class_delete(events):
    for msg in events:
        if msg[-1].strip() == "c2":
            return True
    return False


class OOM:
    def __init__(self) -> None:
        self.t = False
        self.all = 0
        self.tot = 0
        self.classes = set()
class Group:
    def __init__(self):
        self.objs = set()
        self.uses = set()
        self.refinements = set()
class ClassType:
    def __init__(self):
        self.mro = [] # [classname, attrname: P(classname)]
        self.attribute_map = {} # attrname: P(classname)
        self.uses = set()

    def mro_list(self):
        return [x[0] for x in self.mro]

# Three are four kinds of data structures used: 
# OOM: number of objects/classes expose some kind of behaviour, computed along the life circle reconstruction 
# Table 2a, Table 2b, Figure 6a are directly calculated from OOM
# Group: 
# Figure 3a, 3b, Figure 4a, 4b, 4c, Figure 5c, Figure 6b, Table 3
# Runtime:
# Figure 5a, 5b
# ClassType (ClassTable):



# There are three layers of instrumentation
# layer1: instruction
# read events are currently collected in layer1 
# layer2: operator
# layer3: dict
# write events


def hasslots(clas):
    for c, m in global_class_classes[clas].mro:
        if '__slots__' in m:
            return True
    return False
def analyze_attr():

    global E, D, C, U, R, M, M1, M2, M3, M4, M5, M6, M7, V1, V2, V3, V4, P, LL1, LL2, LL3, SL1, SL2, SL3, LD, SD
    global SSS1 , SSS2 , SSS3 , SSS4 , SSS5 , SSS6, SSS7, SSL1 , SSL2 , SSL3 , SSL4 , SSL5, SSL6, SSL7
    global LSS1 , LSS2 , LSS3 , LSL1 , LSL2 , LSL3 
    global DSS1 , DSS2 , DSL1 , DSL2 
    
    global UPD, NON_OVR, OVR, NON_UPD
    global J1, J2, J1_, J2_
    global NNUM, SNUM
    global YES, POLY, GUARDED
    global POLY_CONS, POLY_CONS_HYB, POLY_CONS_TYP, POLY_CONS_EXT, POLY_CONS_PARA, POLY_CONS_LOC, MOD_CLS, DEL_CLS, EXT_CLS, FUN_CLS, FUN_CLS2
    global nob, ob, ln, rn, t1st2, t2st1, t1et2, all_mut, stru

    # ale = attr_load_events()
    # mro = load_mro()
    with open("Loading", "rb") as f:
        Loading = pkl.load(f)
    with open("Setting", "rb") as f:
        Setting = pkl.load(f)
    CLS = defaultdict(dict)
    idxs = set()
    class_objects = {}
    class_objects_meth_sensi = {}
    
    class_objects_cons = {}
    class_objects_cons_para = {}
    class_objects_cons_loc = {}
    
    class_objects_loc_sensi = {}
    class_objects_evo_sensi = {}
    class_objects_para_sensi = {}
    class_objects_para_evo_sensi = {}
    
    dang = defaultdict(int)
    mul = sig = ext = del_ = mod = clr = upd = refl = ove = abn = 0
    poly_cls = 0
    self_ext_set, ext_set = (set() for i in range(2))
    ext_event = 0
    i0 = i1 = i2 = i3 = 0
    r0 = r1 = r2 = r3 = 0
    OOMs = {k: OOM() for k in oom_names}

    extensive_class, deleting_class, clearing_class, replacing_class, modfying_class, updating_class, overriding_class, refl_class = (set() for i in range(8))
    extensive = deleting = clearing = modfying = 0
    nobs, obs, lns, rns, t1st2s, t2st1s, t1et2s, all_muts, strus = (set() for i in range(9))
    nob_ov, ob_ov, ln_ov, rn_ov, t1st2_ov, t2st1_ov, t1et2_ov, all_mut_ov, stru_ov= ([0] for i in range(9))
    nobs_ovs, obs_ovs, lns_ovs, rns_ovs, t1st2s_ovs, t2st1s_ovs, t1et2s_ovs, all_muts_ovs, strus_ovs = (set() for i in range(9))

    nob, ob, ln, rn, t1st2, t2st1, t1et2, all_mut, stru = ([0] for i in range(9))
   # nob = ob = ln = rn = t1st2 = t2st1 = t1et2 = mutative = 0
    # nobs_ov, obs_ov, lns_ov, rns_ov, t1st2s_ov, t2st1s_ov, t1et2s_ov, all_muts_ov = ([0] for i in range(8))
    exp1, imp_mem1, imp_func1, imp_ali1 = ([0] for i in range(4))
    exp2, imp_mem2, imp_func2, imp_ali2 = ([0] for i in range(4))
    exp1_s, imp_mem1_s, imp_func1_s, imp_ali1_s = (set() for i in range(4))
    exp2_s, imp_mem2_s, imp_func2_s, imp_ali2_s = (set() for i in range(4))

    
    # for key in DOP:
    #     DOP[key].append(0)
    # for key in FUN:
    #     FUN[key].append(0)
    refcnts = defaultdict(int)
    maxrefcnts = defaultdict(int)
    Vt = defaultdict(dict)
    attr_type_change = defaultdict(int)
    rp = defaultdict(int)
    mro = frames["store_attr"][0]
    cls_elv = frames["store_attr"][1]
    object_dicts = frames["store_attr"][2]
    for clas in mro:
        global_class_classes[clas] = ClassType()
        global_class_classes[clas].mro = mro_class(global_class_classes, mro, mro[clas])
    for clas in mro:
        mro_type_write(global_class_classes, mro, mro[clas])
    
    cnts = defaultdict(int)
    i = 0
    Loading = defaultdict(dict)
    Setting = defaultdict(dict)

    descrs_get = defaultdict(int)
    descrs_set = defaultdict(int)
    descrs_del = defaultdict(int)
    
    descrs_gets = defaultdict(set)
    descrs_sets = defaultdict(set)
    descrs_dels = defaultdict(set)
    Parameteric_Map = {}
    Location_Map = {}
    S1 = S2 = S3 = S4 = S5 = S6 = S7 = 0
    L1 = L2 = L3 = 0
    D1 = D2 = 0
    S1s, S2s, S3s, S4s, S5s, S6s, S7s = (set() for i in range(7))
    L1s, L2s, L3s = (set() for i in range(3))
    D1s, D2s = (set() for i in range(2))
    obj_cnts = 0 
    non_over = over = 0
    non_update = update = 0
    ccc = set()
    VT = defaultdict(list)
    # error_object = read_error_object()
    for obj in tqdm(object_dicts):
        i += 1
        # if i < 80000:
        #     continue
        # if i > 10000:
        #     break
            
        if obj.find('(nil):0') != -1:
            continue
        if obj.find('builtins') != -1:
            continue
        if obj.find('tests.') != -1:
            continue
        if obj.find('test.') != -1:
            continue
        # a very large objects may cause the analysis to oom
        # if len(object_dicts[obj]) > 300000:
        #     continue
        # if obj.find('sklearn.decomposition._dict_learning.DictionaryLearning') == -1:
        #     continue
        # if obj.find('pyro.distributions.transforms.affine_autoregressive.AffineAutoregressive') == -1:
        #     continue
        
        if obj.find('unknown_module') != -1:
            assert False
        
        initialize_fail = not any(info.find('leaving-constructor') != -1 for _, info in object_dicts[obj])

        if initialize_fail:
            continue
        if repo == 'sklearn':
            if any('leaving-constructor' in info and 'test_estimators_do_not_raise_errors_in_init_or_set_param' in info for _, info in object_dicts[obj]):
                continue
            if any('leaving-constructor' in info and 'check_set_params' in info for _, info in object_dicts[obj]):
                continue
            
        
        clas = obj.split(":")[0].split('-')[0]
        # if clas != 'snorkel.analysis.scorer.Scorer':
        #     continue
        if repo == 'yapf':
            if clas.strip() == 'yapf.yapflib.format_token.FormatToken':
                continue
        if repo == 'faker':
            if clas.strip() == 'faker.generator.Generator':
                continue
            # only count provider once
            if 'Provider' in clas and clas != 'BaseProvider':
                continue
        if clas not in global_class_classes:
            continue

        if hasslots(clas):
            continue
        if clas not in Parameteric_Map:
            Parameteric_Map[clas] = defaultdict(set)
            Location_Map[clas] = defaultdict(set)
            class_objects[clas] = defaultdict(Group)
            class_objects_cons[clas] = defaultdict(Group)
            class_objects_cons_para[clas] = defaultdict(Group)
            class_objects_cons_loc[clas] = [defaultdict(Group), defaultdict(Group), defaultdict(Group), defaultdict(Group), defaultdict(Group)]
            
            class_objects_loc_sensi[clas] = [defaultdict(Group), defaultdict(Group), defaultdict(Group), defaultdict(Group), defaultdict(Group)]
            class_objects_evo_sensi[clas] = defaultdict(Group)
            class_objects_para_evo_sensi[clas] = defaultdict(Group)
            class_objects_para_sensi[clas] = defaultdict(Group)
            
            class_objects_meth_sensi[clas] = defaultdict(Group)
            
        obj_cnts += 1
        vt = defaultdict(set)
        for name in oom_names:
            OOMs[name].t = False
            OOMs[name].tot += 1
        extensive = 0
        deleting = 0
        clearing = 0
        modfying = 0
        overriding = 0
        refing = 0
        reping = 0
        dangerous_ref = defaultdict(int)
        type_env = {}
        type_invariant = {}
        type_monotonic = {}
        
        refcnt = 0
        maxrefcnt = 0
        dict_updating = 0

        initializing = True

        loc_sensitivity_string = [clas, clas, clas, clas, clas]
        evo_sensitivity_string = clas
        para_sensitivity_string = clas
        para_evo_sensitivity_string = clas
        
        meth_sensitivity_string = clas
        cons_loc = None
        evo_step = 0
        test_func = None
        
        for global_id, info in object_dicts[obj]:
            if info == 'init dict':
                type_env.clear()

            elif info == 'assign dict':
                type_env.clear()
                reping = 1
            
            elif info.find('leaving-constructor') != -1:
                initializing = False
                # analyze the parameters rich.containers.Lines-0x153c0c9f3be0:<(nil)>: leaving-constructor: (builtins.list,): 
                msg = info.split(":")
                loc = msg[3].strip()
                # loc = tuple(loc.split('|')[:1])
                cons_loc = loc.split('|')[0]
                tup_part = msg[4]
                dict_part = msg[5]
                # Parameteric_Map[clas][(tup_part, dict_part)].add(frozenset(type_env.items()))
                # Location_Map[clas][loc].add(frozenset(type_env.items()))
                
                for i in range(LOC_K):
                    if i > 0:
                        ctx_len = 2*i-1 
                        loc_sensitivity_string[i] += '|' + str(tuple(loc.split('|')[:ctx_len]))
                    
                    class_objects_loc_sensi[clas][i][loc_sensitivity_string[i]].objs.add(frozenset(type_env.items()))
                    class_objects_cons_loc[clas][i][loc_sensitivity_string[i]].objs.add(frozenset(type_env.items()))
                    
                
                find_test = sum('-test_' in fra for fra in loc.split('|'))
                if find_test:
                    for fra in loc.split('|'):
                        if '-test_' in fra:
                            test_func = fra

                if find_test and test_func:
                    l =  test_func.strip().split('-')[-1]
                    func = test_func.strip().split('-')[-2]
                    f = '-'.join(test_func.strip().split('-')[:-2])
                    test_func = f + '/' + func
                    if os.path.exists(f):
                        func_node = get_func_node(f, func, l)
                        if func_node is not None and len(func_node.args.args) == 0:
                            for i in range(LOC_K):
                                if loc_sensitivity_string[i] not in con_objs[i][f + '/' + func]:
                                    con_objs[i][f + '/' + func][loc_sensitivity_string[i]] = {}
                                con_objs[i][f + '/' + func][loc_sensitivity_string[i]][obj] = [global_id, []]
                        
                para_evo_sensitivity_string += '|' + str((tup_part, dict_part))
                para_sensitivity_string += '|' + str((tup_part, dict_part))
                
                class_objects_evo_sensi[clas][evo_sensitivity_string].objs.add(frozenset(type_env.items()))
                class_objects_para_sensi[clas][para_sensitivity_string].objs.add(frozenset(type_env.items()))
                class_objects_para_evo_sensi[clas][para_evo_sensitivity_string].objs.add(frozenset(type_env.items()))
                
                meth_sensitivity_string += '|' + str((tup_part, dict_part))
                class_objects[clas][clas].objs.add(frozenset(type_env.items()))
                class_objects_meth_sensi[clas][meth_sensitivity_string].objs.add(frozenset(type_env.items()))
                
                class_objects_cons[clas][clas].objs.add(frozenset(type_env.items()))
                class_objects_cons_para[clas][clas + '|' + str((tup_part, dict_part))].objs.add(frozenset(type_env.items()))
                
                for k, v in type_env.items():
                    vt[k].add(v)
                
            else:
                msg = info.split(":")
                if len(msg) == 10:
                    pass
                    # layer 1
                    # if msg[8].find('loading') != -1:
                    #     identity = msg[2] # loc-based dup
                    #     if msg[8].strip() == 'loading 1':
                    #         L1 += 1
                    #         find_len, all_len = member_para(msg[4])
                    #         find_len += 1
                    #         if find_len > 1:
                    #             if find_len == all_len:
                    #                 inherited_descr[0] += 1
                    #             # inherited_repo[repo][msg[5].strip()] += 1
                    #             else:
                    #                 inherited_descr[1] += 1
                    #         DEPTH_DESCR.append(find_len)
                    #         DEPTH_DESCR_LEN[find_len] += 1
                    #         DEPTH_ATTR_REPO[repo].append(find_len)
                    #         if msg[7].strip() == 'builtins.getset_descriptor':
                    #             GETSET.add(msg[5].strip())
                    #         descrs_get[msg[7].strip()] += 1
                    #         L1s.add(identity)
                    #         descrs_gets[msg[7].strip()].add(identity)

                    #         OOMs['get-descr'].t = True
                    #     elif msg[8].strip() == 'loading 2':
                    #         L2 += 1
                    #         find_len, all_len = member_para(msg[4])
                    #         find_len += 1
                    #         if find_len > 1:
                    #             if find_len == all_len:
                    #                 inherited[0] += 1
                    #             else:
                    #                 inherited[1] += 1
                    #             # inherited_repo[repo][msg[5].strip()] += 1
                    #         DEPTH_ATTR.append(find_len)
                    #         DEPTH_ATTR_LEN[find_len] += 1
                    #         L2s.add(identity)

                    #         OOMs['inherit'].t = True
                    #     else:
                    #         L3 += 1
                    #         L3s.add(identity)
                if len(msg) == 9:
                    # layer1
                    if msg[7].find('deling') != -1:
                        identity = msg[2] # loc-based dup
                        if msg[7].strip() == 'deling 1':
                            D1 += 1
                            descrs_del[msg[6].strip()] += 1
                            D1s.add(identity)
                            descrs_dels[msg[6].strip()].add(identity)

                            OOMs['del-descr'].t = True
                        elif msg[7].strip() == 'deling 2':
                            D2 += 1
                            D2s.add(identity)
                            
                    elif msg[7].find('setting') != -1:
                        identity = msg[2] # loc-based dup
                        if msg[7].strip() == 'setting 1':
                            S1 += 1
                            descrs_set[msg[6].strip()] += 1
                            S1s.add(identity)
                            descrs_sets[msg[6].strip()].add(identity)
                            OOMs['set-descr'].t = True
                        elif msg[4].strip() in type_env:
                            if not_same_class(msg[5].strip(), type_env[msg[4].strip()]):
                                # update_m
                                S6 += 1
                                S6s.add(identity)
                            else:
                                # update
                                S2 += 1
                                S2s.add(identity)
                            # 

                        elif msg[7].strip() == 'setting 3':
                            # override
                            # override_m 

                            # add the class-level type
                            if not not_same_class(msg[5], msg[6]):
                                non_over += 1
                                S3 += 1
                                S3s.add(identity)
                            else:
                                over += 1
                                # sub = type_change_pattern(msg[6], msg[5], t1et2_ov, t1et2s_ovs, t1st2_ov, t1st2s_ovs, t2st1_ov, t2st1s_ovs, nob_ov, nobs_ovs, ob_ov, obs_ovs, ln_ov, lns_ovs, rn_ov, rns_ovs, all_mut_ov, all_muts_ovs, stru_ov, strus_ovs, 'override', msg[2])
                                # OOMs['override'].t = True
                                S7 += 1
                                S7s.add(identity)
                        else:
                            member_method, para = member_para(msg[3])
                            # if msg[2].find('__init__') != -1 and member_method and para!= -1:
                            if initializing:
                                S5 += 1
                                S5s.add(identity)
                            else:
                                S4 += 1
                                S4s.add(identity)
                                
                                ext_set.add(msg[2])
                                

                    else:
                        # layer 1
                        name, attr, t = msg[1].strip(), msg[4].strip(), str(extract_nominal(msg[5].strip()))
                        if msg[6].strip() == 'insert 0':
                            i0 += 1
                        elif msg[6].strip() == 'insert 1':
                            i1 += 1
                        elif msg[6].strip().find('insert 2') != -1:
                            operation = msg[6][9:].strip()
                            # DOP[operation][-1] += 1
                            # union, replacement, update
                            i2 += 1
                        else:
                            # func
                            refing = 1
                            func = msg[6][9:].strip()
                            # FUN[func][-1] += 1
                            i3 += 1
                        identity = msg[2] # clas + '.' + attr
                        if initializing:
                            local_evo = 1
                        else:
                            local_evo = int('/'.join(msg[2].split('-')[:-1]).strip() == '/'.join(cons_loc.split('-')[:-1]).strip())
                        member_method, para = member_para(msg[3])
                        mono = 0

                        if not initializing: 
                            if attr in type_env:
                                # if t != type_invariant[attr]:
                                #     sub2, mono2 = type_change_pattern(type_invariant[attr], t, t1et2, t1et2s, t1st2_ov, t1st2s, t2st1, t2st1s, nob, nobs, ob, obs, ln, lns, rn, rns, all_mut, all_muts, stru, strus, 'inv', identity, global_class_classes)
                                #     if not sub2:
                                #         OOMs['abnormal'].t = True
                                #         OOMs['abextend'].t = True
                                if t != type_monotonic[attr]:
                                    sub2, mono2 = type_change_pattern(type_monotonic[attr], t, t1et2, t1et2s, t1st2_ov, t1st2s, t2st1, t2st1s, nob, nobs, ob, obs, ln, lns, rn, rns, all_mut, all_muts, stru, strus, 'mono', identity, global_class_classes)
                                    if not sub2:
                                        OOMs['abmonotonic'].t = True
                                    else:
                                        type_monotonic[attr] = t
                                        mono = 1
                                if t != type_env[attr]: # not_same_class(t, type_env[attr]):
                                    non_update += 1
                                    if msg[6].strip() == 'insert 1':
                                        alias_pattern(msg[3], msg[2], imp_mem1, imp_mem1_s, imp_func1, imp_func1_s, imp_ali1, imp_ali1_s, exp1, exp1_s, int(msg[7]), local_evo, 'evolve')    
                                        loc = msg[2]
                                        l =  loc.strip().split('-')[-1]
                                        func = loc.strip().split('-')[-2]
                                        f = '-'.join(loc.strip().split('-')[:-2])
                                        with open(f) as fi:
                                            ll = fi.read().splitlines()[int(l)-1]
                                        wp, name = get_wp(f, func, l, attr)
                                        wp2, name2 = get_wp_only_cond(f, func, l, attr)
                                        
                                        # if wp and ll.count(attr) == 1:
                                        #     if not isinstance(wp, ast.Constant):
                                        #         branch = branch_check(f, func, l, attr)
                                        #         if branch:
                                        #             ALLCOND.add(loc)
                                        #         else:
                                        #             swp = Simplifier().visit(wp)
                                        #             cnf_wp = cnf_expr(swp)
                                        #             values = _expand(ast.And, cnf_wp)
                                        #             cnf_wp = ast.BoolOp(ast.And(), values)
                                        #             hasa = checked_condition(name, attr, cnf_wp)
                                        #             if isinstance(wp2, ast.Constant):
                                        #                 OTHERCOND2.add(loc)
                                        #             else:
                                        #                 if local_evo:
                                        #                     OTHERCOND1.add(loc)
                                        #                 else:
                                        #                     OTHERCOND3.add(loc)
                                        #                 # elif isinstance(wp2, ast.Constant):
                                        #                 #     OTHERCOND2.add(loc)
                                        #                 # else:
                                                        
                                        #     else:
                                        #         UNCOND.add(loc)
                                    # modification
                                    for ds in DS:
                                        if refcnt > ds:
                                            dangerous_ref[ds] = 1 
                                    OOMs['update'].t = True
                                    OOMs['evolve'].t = True

                                    if int(msg[7]) == 0:
                                        OOMs['inlinear-value'].t = True
                                    if test_func:
                                        for i in range(LOC_K):
                                            if loc_sensitivity_string[i] in con_objs[i][test_func] and obj in con_objs[i][test_func][loc_sensitivity_string[i]]:
                                                con_objs[i][test_func][loc_sensitivity_string[i]][obj][1].append((global_id, info))
                                    # all_mut[0] += 1
                                    # all_muts.add((type_env[attr], t))
                                    attr_type_change[(type_env[attr], t)] += 1 
                                    
                                    sub1, mono1 = type_change_pattern(type_env[attr], t, t1et2, t1et2s, t1st2_ov, t1st2s, t2st1, t2st1s, nob, nobs, ob, obs, ln, lns, rn, rns, all_mut, all_muts, stru, strus, 'update', identity, global_class_classes)
                                else:
                                    update += 1
                            else:
                                
                                mono = 1
                                vt[attr].add('missing')
                                # add_type_self(global_class_classes, mro, clas, attr, "missing")
                                # else:
                                #     if msg[8].find('property') == -1:
                                #         add_type(clas, vt, attr, msg[8].strip())
                                        
                                ccc.add(clas + '.' + attr)
                                OOMs['extend'].t = True
                                OOMs['evolve'].t = True
                                if test_func:
                                    for i in range(LOC_K):
                                        if loc_sensitivity_string[i] in con_objs[i][test_func] and obj in con_objs[i][test_func][loc_sensitivity_string[i]]:
                                            con_objs[i][test_func][loc_sensitivity_string[i]][obj][1].append((global_id, info))
                                OOMs['abnormal'].t = True
                                if int(msg[7]) == 0:
                                    OOMs['inlinear-value'].t = True
                                if msg[6].strip() == 'insert 1':

                                    
                                    alias_pattern(msg[3], msg[2], imp_mem1, imp_mem1_s, imp_func1, imp_func1_s, imp_ali1, imp_ali1_s, exp1, exp1_s, int(msg[7]), local_evo, 'evolve')
                                    loc = msg[2]
                                    l =  loc.strip().split('-')[-1]
                                    func = loc.strip().split('-')[-2]
                                    f = '-'.join(loc.strip().split('-')[:-2])
                                    with open(f) as fi:
                                        ll = fi.read().splitlines()[int(l)-1]
                                    wp, name = get_wp(f, func, l, attr)
                                    wp2, name2 = get_wp_only_cond(f, func, l, attr)
                                    if wp and ll.count(attr) == 1:
                                        if not isinstance(wp, ast.Constant):
                                            branch = branch_check(f, func, l, attr)
                                            if branch:
                                                ALLCOND.add(loc)
                                            else:
                                                swp = Simplifier().visit(wp)
                                                cnf_wp = cnf_expr(swp)
                                                values = _expand(ast.And, cnf_wp)
                                                cnf_wp = ast.BoolOp(ast.And(), values)
                                                hasa = checked_condition(name, attr, cnf_wp)
                                                if isinstance(wp2, ast.Constant):
                                                    OTHERCOND2.add(loc)
                                                else:
                                                    if local_evo:
                                                        OTHERCOND1.add(loc)
                                                    # elif isinstance(wp2, ast.Constant):
                                                    #     OTHERCOND2.add(loc)
                                                    else:
                                                        OTHERCOND3.add(loc)
                                        else:
                                            UNCOND.add(loc)
                        # object-class
                        if msg[8].strip() != "NULL":
                            if not_same_class(t,  msg[8].strip()):
                                # override
                                sub, mono = type_change_pattern(msg[8].strip(), t, t1et2_ov, t1et2s_ovs, t1st2_ov, t1st2s_ovs, t2st1_ov, t2st1s_ovs, nob_ov, nobs_ovs, ob_ov, obs_ovs, ln_ov, lns_ovs, rn_ov, rns_ovs, all_mut_ov, all_muts_ovs, stru_ov, strus_ovs, 'override', identity, global_class_classes)
                                OOMs['override'].t = True
                            elif t in POSSIBLE_FUNC:
                                OOMs['object_func'].t = True
                        if attr not in type_env:
                            type_invariant[attr] = t
                            type_monotonic[attr] = t
                        type_env[attr] = t

                        if msg[8].strip() == "NULL":
                            CLS[clas][attr] = 'missing'
                        else:
                            CLS[clas][attr] = msg[8].strip()

                        if not initializing:
                            for i in range(LOC_K):
                                class_objects_loc_sensi[clas][i][loc_sensitivity_string[i]].objs.add(frozenset(type_env.items()))
                            evo_step += 1
                            if mono == 1:
                                para_evo_sensitivity_string += '|' + str(attr) + ' : ' + str(t)
                                evo_sensitivity_string += '|' + str(attr) + ' : ' + str(t)
                            class_objects_para_sensi[clas][para_sensitivity_string].objs.add(frozenset(type_env.items()))
                            class_objects_evo_sensi[clas][evo_sensitivity_string].objs.add(frozenset(type_env.items()))
                            class_objects_para_evo_sensi[clas][para_evo_sensitivity_string].objs.add(frozenset(type_env.items()))
                            
                            if (para and member_method) or local_evo:
                                if msg[2].strip() not in meth_sensitivity_string:
                                    meth_sensitivity_string += '|' + msg[2].strip()
                            class_objects[clas][clas].objs.add(frozenset(type_env.items()))
                            class_objects_meth_sensi[clas][meth_sensitivity_string].objs.add(frozenset(type_env.items()))
                            
                            vt[attr].add(t)
                        class_objects[clas][clas].refinements.add((msg[2], attr, t))

                        class_objects_para_sensi[clas][para_sensitivity_string].refinements.add((msg[2], attr, t))
                        class_objects_evo_sensi[clas][evo_sensitivity_string].refinements.add((msg[2], attr, t))
                        class_objects_para_evo_sensi[clas][para_evo_sensitivity_string].refinements.add((msg[2], attr, t))
                        for i in range(LOC_K):
                            class_objects_loc_sensi[clas][i][loc_sensitivity_string[i]].refinements.add((msg[2], attr, t))
                        add_type_self(global_class_classes, mro, clas, attr, t)
                        add_type_write(global_class_classes, mro, repr_to_obj(msg[5].strip()))
                if len(msg) == 8:
                    pass
                    
                elif len(msg) == 7:
                    # operator-level get/set
                    print(msg)

                    assert False
                elif len(msg) == 6:
                    # simplified layer-1 load
                    if msg[5].find('-') != -1:
                        # load
                        if msg[5].find('builtins.method') == -1:
                            class_objects[clas][clas].uses.add((msg[2], msg[4]))
                            for i in range(LOC_K):
                                class_objects_loc_sensi[clas][i][loc_sensitivity_string[i]].uses.add((msg[2], msg[4]))
                            class_objects_evo_sensi[clas][evo_sensitivity_string].uses.add((msg[2], msg[4]))
                            class_objects_para_sensi[clas][para_sensitivity_string].uses.add((msg[2], msg[4]))
                            class_objects_para_evo_sensi[clas][para_evo_sensitivity_string].uses.add((msg[2], msg[4]))
                            if clas in global_class_classes:
                                global_class_classes[clas].uses.add((msg[2], msg[4]))
                            # pass

                    # dict-level: delete or loading
                    elif msg[5].find('read') != -1:
                        if msg[5].strip() == 'read 0':
                            r0 += 1
                        elif msg[5].strip() == 'read 1':
                            r1 += 1
                        elif msg[5].strip() == 'read 2':
                            r2 += 1
                        else:
                            func = msg[5][6:]
                            F2[func] += 1
                            if func == 'getattr':
                                S.add(msg[2])
                            r3 += 1
                    else:
                        # layer2 delete
                        attr = msg[4].strip()
                        if attr in type_env:
                            del type_env[attr]
                        vt[attr].add('missing')
                        deleting = 1
                        OOMs['remove'].t = True
                        OOMs['evolve'].t = True
                        
                        OOMs['abnormal'].t = True
                        OOMs['abextend'].t = True
                        OOMs['abmonotonic'].t = True
                        DELETE_INFOS.add(info)
                        for ds in DS:
                            if refcnt > ds:
                                dangerous_ref[ds] = 1 
                elif len(msg) == 5:
                    # dict-level: clear/merge
                    if msg[4].strip() == 'clear':
                        type_env.clear()
                        clearing = 1
                    else:
                        dict_updating = 1

                    # for ds in DS:
                    #     if refcnt > ds:
                    #         dangerous_ref[ds] = 1 

                elif len(msg) == 4:
                    # refcnt
                    pass
                    # if msg[0].strip() == 'overloading':
                    #     pass

                    # else:
                    #     if msg[3].strip().find('increase') != -1:
                    #         refcnt += 1
                    #     elif msg[3].find('decrease') != -1:
                    #         refcnt -= 1
                    #     maxrefcnt = max(maxrefcnt, refcnt)
        # if any(len(vt[attr]) > 1 for attr in vt):
        #     OOMs['evolve'].t = True
        # if any(len(vt[attr]) > 0 for attr in vt):
        #     OOMs['tem_evolve'].t = True
        if OOMs['extend'].t or OOMs['update'].t or OOMs['remove'].t:
            OOMs['evolve'].t = True
            if OOMs['extend'].t and OOMs['update'].t:
                OOMs['both'].t = True
            elif OOMs['extend'].t:
                OOMs['only-e'].t = True
            elif OOMs['update'].t:
                OOMs['only-u'].t = True
        for name in OOMs:
            m = OOMs[name]
            if m.t:
                m.classes.add(clas)
                m.all += 1
        
    
    with open(f'save_inter/global_class_classes_{repo}', 'wb') as f:
        pkl.dump(global_class_classes, f)
    with open(f'save_inter/class_objects_{repo}', 'wb') as f:
        pkl.dump(class_objects, f)
    with open(f'save_inter/class_objects_cons_{repo}', 'wb') as f:
        pkl.dump(class_objects_cons, f)
    with open(f'save_inter/class_objects_cons_para_{repo}', 'wb') as f:
        pkl.dump(class_objects_cons_para, f)
    with open(f'save_inter/class_objects_cons_loc_{repo}', 'wb') as f:
        pkl.dump(class_objects_cons_loc, f)
    
    with open(f'save_inter/class_objects_loc_sensi_{repo}', 'wb') as f:
        pkl.dump(class_objects_loc_sensi, f)
    with open(f'save_inter/class_objects_evo_sensi_{repo}', 'wb') as f:
        pkl.dump(class_objects_evo_sensi, f)
    with open(f'save_inter/class_objects_para_sensi_{repo}', 'wb') as f:
        pkl.dump(class_objects_para_sensi, f)
    with open(f'save_inter/class_objects_para_evo_sensi_{repo}', 'wb') as f:
        pkl.dump(class_objects_para_evo_sensi, f)
    
    with open(f'save_inter/class_objects_meth_sensi_{repo}', 'wb') as f:
        pkl.dump(class_objects_meth_sensi, f)
    with open(f'save_inter/CPATTERN_DUP_{repo}', 'wb') as f:
        pkl.dump(CPATTERN_DUP, f)
    with open(f'save_inter/APATTERN_DUP_{repo}', 'wb') as f:
        pkl.dump(APATTERN_DUP, f)
    with open(f'save_inter/PAT2_{repo}', 'wb') as f:
        pkl.dump(PAT2, f)

    
    with open(f'save_inter/UNCOND_{repo}', 'wb') as f:
        pkl.dump(UNCOND, f)
    with open(f'save_inter/ALLCOND_{repo}', 'wb') as f:
        pkl.dump(ALLCOND, f)
    with open(f'save_inter/HASCOND_{repo}', 'wb') as f:
        pkl.dump(HASCOND, f)
    with open(f'save_inter/OTHERCOND1_{repo}', 'wb') as f:
        pkl.dump(OTHERCOND1, f)
    with open(f'save_inter/OTHERCOND2_{repo}', 'wb') as f:
        pkl.dump(OTHERCOND2, f)
    with open(f'save_inter/OTHERCOND3_{repo}', 'wb') as f:
        pkl.dump(OTHERCOND3, f)
    
    
    with open(f'save_inter/OOMs_{repo}', 'wb') as f:
        pkl.dump(OOMs, f)

    with open(f'save_inter/con_objs_{repo}', 'wb') as f:
        pkl.dump(con_objs, f)

    with open(f'save_inter/class_objs_{repo}', 'wb') as f:
        pkl.dump(class_objs, f)

    
    return class_objects_meth_sensi, global_class_classes, class_objects, class_objects_cons, class_objects_cons_para, class_objects_cons_loc, class_objects_loc_sensi, class_objects_evo_sensi, class_objects_para_sensi, class_objects_para_evo_sensi, OOMs




def result_report(cls_elv, class_objects_meth_sensi, global_class_classes, class_objects, class_objects_cons, class_objects_cons_para, class_objects_cons_loc, class_objects_loc_sensi, class_objects_evo_sensi, class_objects_para_sensi, class_objects_para_evo_sensi, OOMs, con_objs):
    global YES, POLY, GUARDED
    global POLY_CONS, POLY_CONS_HYB, POLY_CONS_TYP, POLY_CONS_EXT, POLY_CONS_PARA, POLY_CONS_LOC, MOD_CLS, DEL_CLS, EXT_CLS, FUN_CLS, FUN_CLS2
    global METH_POLY_CLS, POLY_CLS
    global SIM_ATTR, STRU_ATTR, COER_ATTR, OTHER_ATTR, NATTR, EATTR
    global SIM_ATTR2, STRU_ATTR2, COER_ATTR2, OTHER_ATTR2, NATTR2, EATTR2, NNEATTR2, POLY_CONS2, POLY_HYB, STA_CLS
    global POLY_CONS_HYB2, POLY_CONS_TYP2, POLY_CONS_EXT2
    
    tra = 0 
    non_tra = 0
    no_wp = 0
    yes_wp = 0
    trival_not_wp = 0
    trival_wp = 0
    guarded = 0
    union_cover1 = 0
    manual_check1 = 0
    union_cover2 = 0
    manual_check2 = 0
    # num = min(len(possible_attrs), 15)
    # NNUM += num
    # SAM = random.sample(possible_attrs, num)
    # PALE = dict(sorted(PALE.items(), key=lambda x: x[0]))
    
    def same_func_before(loc1, loc2):
        func1 = loc1.strip().split('-')[-2]
        func2 = loc2.strip().split('-')[-2]
        l1 =  loc1.strip().split('-')[-1]
        l2 =  loc2.strip().split('-')[-1]
        return func1 == func2 and l2 <= l1
    def type_evaluator( all_type, loc, use_filter=True, nominal=False, aggressive_guards=False):
        l =  loc.strip().split('-')[-1]
        func = loc.strip().split('-')[-2]
        f = '-'.join(loc.strip().split('-')[:-2])
        if os.path.exists(f) and f.find(repo) != -1: # and f.find('test') == -1:
            # remove thrid party and tests
            usage = get_usage(f, func, l, attr)
            with open(f) as fi:
                ll = fi.read().splitlines()[int(l)-1]
            if usage:
                if use_filter:
                    wp, name = get_wp(f, func, l, attr)
                    if wp and name:
                        if not isinstance(wp, ast.Constant):
                            swp = Simplifier().visit(wp)
                            cnf_wp = cnf_expr(swp)
                            values = _expand(ast.And, cnf_wp)
                            cnf_wp = ast.BoolOp(ast.And(), values)
                            reachable_type = filter_type(all_type, name, attr, cnf_wp)
                        else:
                            reachable_type = all_type
                    else:
                        reachable_type = all_type
                    
                    if name:
                        reachable_type = filter_type_expr(reachable_type, name, attr, ll)
                        if aggressive_guards:
                            try:
                                if name in pprint2.pprint_top(wp) and attr in pprint2.pprint_top(wp):
                                    return True
                            except Exception:
                                return False
                else:
                    reachable_type = all_type
                succ = find_attr(reachable_type, usage, nominal)
                
                return succ
            else:
                return None
        else:
            return None

    def nominal_supertype(a, b):
        mro = set(global_class_classes[b].mro_list())
        return a in mro
    def nominal_join(a, b):
        if nominal_supertype(a, b):
            return a
        elif nominal_supertype(b, a):
            return b
        else:
            mro1 = set(global_class_classes[a].mro_list())
            mro2 = set(global_class_classes[b].mro_list())
            mro1 = mro1.intersection(mro2)
            if len(mro1) == 0:
                mro1.add('builtins.object')
            assert len(mro1) > 0
            mro1 = list(mro1)
            best = mro1[0]
            for c in mro1:
                if len(global_class_classes[c].mro_list()) > len(global_class_classes[best].mro_list()):
                    best = c
            return best
    def generalized_nominal_join(T):
        assert len(T) >= 1
        base = T[0]
        for t in T[1:]:
            base = nominal_join(base, t)
        return base

    def make_VT(groups, refine = True, loc = False):
        new_Vt = defaultdict(list)
        attr_uses = defaultdict(list)
        refinements = defaultdict(list)
        for group_id in groups:
            # if loc and '|' in group_id and con_objs[group_id] <= 1:
            #     continue
            objs = groups[group_id].objs
            uses = groups[group_id].uses
            refs = groups[group_id].refinements
            for attr in all_attributes:
                new_Vt[attr].append({str(extract_nominal(dict(obj)[attr])) for obj in objs if attr in dict(obj)})
                if int(any(attr not in dict(obj) for obj in objs) and any(attr in dict(obj) for obj in objs) and not inclass(attr, global_class_classes[clas].mro)):
                    new_Vt[attr][-1].add('missing')
                # if attr in class_attributes:
                #     new_Vt2[attr][-1].update(class_attributes[attr])
                attr_uses[attr].append(set())
                for loc, attr2 in uses:
                    attr2 = attr2.split('.')[-1]
                    if attr2 == attr:
                        attr_uses[attr][-1].add(loc)
                if refine:
                    refinements[attr].append(set())
                    for loc, attr2, tt in refs:
                        attr2 = attr2.split('.')[-1]
                        if attr2 == attr:
                            refinements[attr][-1].add((loc, tt))
        if refine:
            return new_Vt, attr_uses, refinements
        else:
            return new_Vt, attr_uses
    # are there any pattern? 

    # attr_type_change
    # construct global_class_classes (only missing is constructed here, for now)
    for clas in global_class_classes:
        if 'test' in clas or 'builtin' in clas:
                continue
        if repo in clas and clas in class_objects:
            objs = class_objects[clas][clas].objs
            for attr in global_class_classes[clas].attribute_map:
                if int(any(attr not in dict(obj) for obj in objs) and not inclass(attr, global_class_classes[clas].mro)):
                    global_class_classes[clas].attribute_map[attr].add('missing')

    
    elvo_clas = copy(OOMs['extend'].classes).union(OOMs['update'].classes)
    
    OOMs['both'].classes.clear()
    OOMs['only-e'].classes.clear()
    OOMs['only-u'].classes.clear()
    
    for clas in elvo_clas:
        if clas in OOMs['extend'].classes and clas in OOMs['update'].classes:
            OOMs['both'].classes.add(clas)
        elif clas in OOMs['extend'].classes:
            OOMs['only-e'].classes.add(clas)
        elif clas in OOMs['update'].classes:
            OOMs['only-u'].classes.add(clas)
    for name in OM:
        ALL_OOM[name][0] += OOMs[name].all
        ALL_OOM[name][1] += OOMs[name].tot
        ALL_OOM[name][2] += len(OOMs[name].classes)
        ALL_OOM[name][3] += len(class_objects)
    for name in oom_names:
        if name != 'evolve':
            ALL_OOMs[name].append((round(div(OOMs[name].all, OOMs['evolve'].all), 3), round(div(len(OOMs[name].classes), len(OOMs['evolve'].classes)), 3)))
        else:
            ALL_OOMs[name].append((round(div(OOMs[name].all, OOMs[name].tot), 3), round(div(len(OOMs[name].classes), len(class_objects)), 3)))
    with open(f'cause/{repo}.txt', 'w+') as ff, open(f'cause_to_examine/{repo}_cons.txt', 'w+') as fr0, open(f'cause_to_examine/{repo}.txt', 'w+') as fr, open(f'class_to_examine/{repo}.txt', 'w+') as frc, open(f'cause/{repo}_???.txt', 'w+') as fm:
        for clas in tqdm(class_objects):
            VV1.append(len(class_objects_cons[clas][clas].objs))
            if len(class_objects_cons[clas][clas].objs) > 1:
                POLY_CONS += 1
                
                new_Vt = {}
                attrs = set()
                objs = class_objects_cons[clas][clas].objs
                for obj in objs:
                    attrs.update({x[0] for x in obj})
                for attr in attrs:
                    new_Vt[attr] = {str(extract_nominal(dict(obj)[attr])) for obj in objs if attr in dict(obj)}
                flag1 = 1
                flag2 = 1
                for obj in objs:
                    obj = dict(obj)
                    for attr in new_Vt:
                        if attr not in obj:
                            DIFF[0] += 1
                            flag1 = 0
                        else:
                            if tuple({str(extract_nominal(obj[attr]))}) != tuple(new_Vt[attr]):
                                DIFF[1] += 1
                                flag2 = 0
                            else:
                                DIFF[2] += 1

                for attr in attrs:
                    if int(any(attr not in dict(obj) for obj in objs) and not inclass(attr, global_class_classes[clas].mro)):
                        new_Vt[attr].add('missing')
                for attr in attrs:
                    if len(new_Vt[attr]) > 1:

                        types = copy(new_Vt[attr])
                        if 'missing' in types:
                            types.remove('missing')
                        if len(types) == 1 :
                            # T ? 
                            EATTR += 1
                        else:
                            if 'builtins.NoneType' in types:
                                types.remove('builtins.NoneType')
                            if len(types) == 1:
                                # Optional[T] after remove ?
                                NATTR += 1
                            else:
                                if generalized_nominal_join2(list(types), global_class_classes):
                                    SIM_ATTR += 1
                                elif generalized_numberic_coercion(list(types)):
                                    COER_ATTR += 1
                                elif generalized_stru_join2(list(types), global_class_classes):
                                    STRU_ATTR += 1
                                else:
                                    OTHER_ATTR += 1
                if flag1 == 0 and flag2 == 0:
                    POLY_CONS_HYB += 1
                elif flag1 ==0:
                    POLY_CONS_EXT += 1
                elif flag2 ==0:
                    POLY_CONS_TYP += 1
            
            if len(class_objects[clas][clas].objs) > 1:
                POLY_CONS2 += 1
                new_Vt = {}
                attrs = set()
                objs = class_objects[clas][clas].objs
                for obj in objs:
                    attrs.update({x[0] for x in obj})
                for attr in attrs:
                    new_Vt[attr] = {str(extract_nominal(dict(obj)[attr])) for obj in objs if attr in dict(obj)}
                flag1 = 1
                flag2 = 1
                for obj in objs:
                    obj = dict(obj)
                    for attr in new_Vt:
                        if attr not in obj:
                            DIFF2[0] += 1
                            flag1 = 0
                        else:
                            if tuple({str(extract_nominal(obj[attr]))}) != tuple(new_Vt[attr]):
                                DIFF2[1] += 1
                                flag2 = 0
                            else:
                                DIFF2[2] += 1
                if flag1 == 0 and flag2 == 0:
                    POLY_CONS_HYB2 += 1
                elif flag1 ==0:
                    POLY_CONS_EXT2 += 1
                elif flag2 ==0:
                    POLY_CONS_TYP2 += 1

            if len(class_objects_cons[clas][clas].objs) > 1 and clas in OOMs['evolve'].classes:
                POLY_HYB += 1
            if len(class_objects_cons[clas][clas].objs) == 1 and clas not in OOMs['evolve'].classes:
                STA_CLS += 1
                frc.write(clas + '\n')
            
            VS1[len(class_objects_cons[clas][clas].objs)] += 1 # degree of constructor poly
            VV2.append(len(class_objects[clas][clas].objs))
            VS2[len(class_objects[clas][clas].objs)] += 1
            if len(class_objects[clas][clas].objs) > 1:
                POLY_CLS += 1
            # Figure 4a
            flag = 0
            for group_id in class_objects_cons_para[clas]:
                objs = class_objects_cons_para[clas][group_id].objs
                if len(objs) > 1 and flag == 0:
                    POLY_CONS_PARA += 1
                    flag = 1
            
            for i in range(LOC_K):
                flag = 0
                for group_id in class_objects_cons_loc[clas][i]:
                    objs = class_objects_cons_loc[clas][i][group_id].objs
                    if len(objs) > 1 and flag == 0:
                        POLY_CONS_LOC[i] += 1
                        flag = 1
            
            func = class_func(cls_elv[clas])
            extend = class_extend(cls_elv[clas])
            evolve = class_modify(cls_elv[clas])
            delete = class_delete(cls_elv[clas])
            if func:
                FUN_CLS += 1
            if extend:
                EXT_CLS += 1
            if evolve:
                MOD_CLS += 1
            if delete:
                DEL_CLS += 1
            # continue
        

            all_attributes = copy(global_class_classes[clas].attribute_map)
            class_attributes = {}
            for c, ca in global_class_classes[clas].mro:
                for k, ct in ca.items():
                    ct = {x for x in ct if 'property' not in x and 'Property' not in x}
                    # if k in all_attributes:
                    #     all_attributes[k].update(ct)
                    # else:
                    #     all_attributes[k] = ct
                    if k in class_attributes:
                        class_attributes[k].update(ct)
                    else:
                        class_attributes[k] = ct
            uses = global_class_classes[clas].uses
            # attr_uses = defaultdict(set)
            # for loc, attr in uses:
            #     attr = attr.split('.')[-1]
            #     attr_uses[attr].add(loc)

            
            new_Vt0, attr_uses0, refinements0 = make_VT(class_objects_cons[clas])
            new_Vt, attr_uses, refinements = make_VT(class_objects[clas])
            new_Vt1 = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
            attr_uses1 = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
            refinements1 = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
            
            for i in range(LOC_K):
                tem1, tem2, tem3 = make_VT(class_objects_loc_sensi[clas][i])
                new_Vt1[i] = tem1
                attr_uses1[i] = tem2
                refinements1[i] = tem3
                
            new_Vt2, attr_uses2, refinements2 = make_VT(class_objects_para_sensi[clas])
            new_Vt3, attr_uses3, refinements3 = make_VT(class_objects_evo_sensi[clas])
            new_Vt4, attr_uses4, refinements4 = make_VT(class_objects_para_evo_sensi[clas])
            
            for attr in all_attributes:
                ALL_ATTR.add(clas + '.' + attr)
                if len(all_attributes[attr]) > 1 and ok(all_attributes[attr]):
                    if 'missing' in new_Vt[attr][0]:
                        if 'missing' in new_Vt0[attr][0]:
                            CONMIS.add(clas + '.' + attr)
                        MIS.add(clas + '.' + attr)
                    POLY_ATTR.add(clas + '.' + attr)


                    types = copy(all_attributes[attr])
                    if 'missing' in types:

                        types.remove('missing')
                    if len(types) == 1 :
                        # T ? 
                        EATTR2 += 1
                    else:
                        if 'builtins.NoneType' in types:
                            types.remove('builtins.NoneType')
                        if len(types) == 1:
                            # Optional[T] after remove ?
                            NATTR2 += 1
                        else:
                            if generalized_nominal_join2(list(types), global_class_classes):
                                SIM_ATTR2 += 1
                            elif generalized_numberic_coercion(list(types)):
                                COER_ATTR2 += 1
                            elif generalized_stru_join2(list(types), global_class_classes):
                                STRU_ATTR2 += 1
                            else:
                                OTHER_ATTR2 += 1
                    # eprint(clas + '.' + attr)
                    # eprint(all_attributes[attr])
                    if len(new_Vt0[attr][0]) > 1:
                        POLY_ATTR0.add(clas + '.' + attr)
                        for loc in attr_uses0[attr][0]:
                            succ0 = type_evaluator(copy(new_Vt0[attr][0]), loc)
                            if succ0 is not None:
                                YES_SITE0.add(clas + '.' + attr)
                            if succ0 == False:
                                GUARDED_SITE0.add(clas + '.' + attr)
                            succ1 = type_evaluator(copy(new_Vt0[attr][0]), loc, False)
                            all_type = copy(new_Vt0[attr][0])
                            if succ1 == False:
                                UNION_SITE0.add(clas + '.' + attr)
                            try:
                                nominal_super = generalized_nominal_join(list(all_type))
                            except KeyError as e:
                                nominal_super = generalized_nominal_join_static(list(all_type))
                            if 'missing' in all_type:
                                nominal_super = 'missing'
                            succ3 = type_evaluator({nominal_super}, loc, nominal = True)
                            if succ3 == False:
                                NOMINAL_ATTR0.add(clas + '.' + attr)

                            refine_types = set()
                            for loc2, t in refinements0[attr][0]:
                                if same_func_before(loc, loc2):
                                    refine_types.add(t)
                            if len(refine_types) == 0:
                                refine_types = copy(new_Vt0[attr][0])
                            succ4 = type_evaluator(refine_types, loc)
                            if succ3 == False:
                                REFINE_SITE0.add(clas + '.' + attr)

                    if len(new_Vt[attr][0]) > 1:
                        POLY_ATTR.add(clas + '.' + attr)
                        for loc in attr_uses[attr][0]:
                            
                            succ0 = type_evaluator(copy(new_Vt[attr][0]), loc)
                            if succ0 is not None:
                                YES_SITE.add(clas + '.' + attr)
                            if succ0 == False:
                                GUARDED_SITE.add(clas + '.' + attr)
                            
                                
                            
                            
                            all_type = copy(new_Vt[attr][0])
                            succ1 = type_evaluator(copy(new_Vt[attr][0]), loc, False)
                            if succ1 == False:
                                UNION_SITE.add(clas + '.' + attr)
                            
                            
                            try:
                                nominal_super = generalized_nominal_join(list(all_type))
                            except KeyError as e:
                                nominal_super = generalized_nominal_join_static(list(all_type))
                            if 'missing' in all_type:
                                nominal_super = 'missing'
                            succ3 = type_evaluator({nominal_super}, loc, nominal = True)
                            if succ3 == False:
                                NOMINAL_ATTR.add(clas + '.' + attr)

                            
                            refine_types = set()
                            for loc2, t in refinements[attr][0]:
                                if same_func_before(loc, loc2):
                                    refine_types.add(t)
                            if len(refine_types) == 0:
                                refine_types = copy(new_Vt[attr][0])
                                types_without_null = remove_null(copy(new_Vt[attr][0]))
                                types_without_none = remove_none(copy(new_Vt[attr][0]))
                                type_without_both = remove_none(remove_null(copy(new_Vt[attr][0])))
                            else:
                                types_without_null = refine_types
                                types_without_none = refine_types
                                type_without_both = refine_types
                            
                            succ00 = type_evaluator(copy(types_without_null), loc)
                            succ01 = type_evaluator(copy(types_without_none), loc)
                            succ02 = type_evaluator(copy(type_without_both), loc)

                            succ_ = type_evaluator(copy(refine_types), loc, aggressive_guards=True)
                            if succ_ == False:
                                GUARDED_SITE_AGGRE.add(clas + '.' + attr)
                            if succ00 == False:
                                CAUSE_ATTR0.add(clas + '.' + attr)
                                
                            if succ01 == False:
                                CAUSE_ATTR1.add(clas + '.' + attr)
                            if succ02 == False:
                                CAUSE_ATTR2.add(clas + '.' + attr)
                            
                            succ4 = type_evaluator(refine_types, loc)

                            if succ00 == True and succ4 == False:
                                fr.write(clas + '.' + attr + '\n')
                                fr.write(str(refine_types) + '\n')
                                fr.write(loc + '\n')
                                l =  loc.strip().split('-')[-1]
                                func = loc.strip().split('-')[-2]
                                f = '-'.join(loc.strip().split('-')[:-2])
                                with open(f) as fi:
                                    ll = fi.read().splitlines()[int(l)-1]
                                fr.write(ll + '\n')
                            # if succ4 == True:
                            #     fr.write(clas + '.' + attr + '\n')
                            #     fr.write(str(refine_types) + '\n')
                            #     fr.write(loc + '\n')
                            #     l =  loc.strip().split('-')[-1]
                            #     func = loc.strip().split('-')[-2]
                            #     f = '-'.join(loc.strip().split('-')[:-2])
                            #     with open(f) as fi:
                            #         ll = fi.read().splitlines()[int(l)-1]
                            #     fr.write(ll + '\n')
                            if succ4 == False:    
                                REFINE_SITE.add(clas + '.' + attr)
                                
                            
                    for i in range(LOC_K):
                        for j, (refs, uses, types) in enumerate(zip(refinements1[i][attr], attr_uses1[i][attr], new_Vt1[i][attr])):
                            for loc in uses:

                                refine_types = set()
                            
                                for loc2, t in refs:
                                    if same_func_before(loc, loc2):
                                        refine_types.add(t)
                                if len(refine_types) == 0:
                                    refine_types = copy(types)
                                succ = type_evaluator(copy(refine_types), loc,)
                                if succ == False:
                                    SENSI_SITE1[i].add(clas + '.' + attr)
                    for j, (refs, uses, types) in enumerate(zip(refinements2[attr], attr_uses2[attr], new_Vt2[attr])):
                        for loc in uses:
                            refine_types = set()
                            
                            for loc2, t in refs:
                                if same_func_before(loc, loc2):
                                    refine_types.add(t)
                            if len(refine_types) == 0:
                                refine_types = copy(types)
                            succ = type_evaluator(copy(refine_types), loc,)
                            if succ == False:
                                SENSI_SITE2.add(clas + '.' + attr)
                    for j, (refs, uses, types) in enumerate(zip(refinements3[attr], attr_uses3[attr], new_Vt3[attr])):
                        for loc in uses:
                            refine_types = set()
                            
                            for loc2, t in refs:
                                if same_func_before(loc, loc2):
                                    refine_types.add(t)
                            if len(refine_types) == 0:
                                refine_types = copy(types)
                            succ = type_evaluator(copy(refine_types), loc,)
                            if succ == False:
                                SENSI_SITE3.add(clas + '.' + attr)
                    for j, (refs, uses, types) in enumerate(zip(refinements4[attr], attr_uses4[attr], new_Vt4[attr])):
                        for loc in uses:
                            refine_types = set()
                            
                            for loc2, t in refs:
                                if same_func_before(loc, loc2):
                                    refine_types.add(t)
                            if len(refine_types) == 0:
                                refine_types = copy(types)
                            succ = type_evaluator(copy(refine_types), loc,)
                            if succ == False:
                                SENSI_SITE4.add(clas + '.' + attr)
                    


    
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
opcodes = defaultdict(list)
infos = defaultdict(list)
mutative = 0
cover = 0
mutative_set = set()
cover_set = set()

def process_repo():
    global class_hierarchy, class_attributes, class_attributes_no_objects,  class_attributes_no_inheritance, c
    
    with open(f"class_tables/ch_{repo}", "rb") as f:
        class_hierarchy = pkl.load(f)
        add_object_back(class_hierarchy)

    with open(f"class_tables/canob_{repo}", "rb") as f:
        class_attributes_no_objects = pkl.load(f)
    with open(f"class_tables/CA", "rb") as f:
        class_attributes = pkl.load(f)
        # for c in class_attributes:
        #     CA[c] = class_attributes[c]
    for k in ['collections.collections.defaultdict', 'requests.structures.CaseInsensitiveDict', 'collections.collections.OrderedDict', 'networkx.classes.coreviews.FilterAtlas', 'networkx.classes.coreviews.FilterMultiAdjacency', 'networkx.classes.coreviews.FilterAdjacency']:
        if k not in class_attributes:
            class_attributes[k] = class_attributes['builtins.dict']
        else:
            class_attributes[k].update(class_attributes['builtins.dict'])

    
    class_attributes['click.types.Tuple'] = class_attributes['click.types.Choice']
    class_attributes['numpy.int32'] = class_attributes['numpy.int64'] = class_attributes['numpy.integer']
    class_attributes['numpy.float32'] = class_attributes['numpy.float64'] = class_attributes['numpy.floating']


    with open(f"class_tables/cani_{repo}", "rb") as f:
        class_attributes_no_inheritance = pkl.load(f)
    
    global frames, opcodes, infos, mutative, cover, mutative_set, cover_set, id_to_master
    global global_class_classes
    global CPATTERN_DUP, APATTERN_DUP, PAT2
    global con_objs, class_objs
    global UNCOND, ALLCOND, HASCOND, OTHERCOND1, OTHERCOND2, OTHERCOND3
    id_to_master = {}
    for c in files:
        # class_analysis()
        opcodes = defaultdict(list)
        infos = defaultdict(list)
        mutative = 0
        cover = 0
        mutative_set = set()
        cover_set = set()
        save_path = f'{repo}_save_{c}' if repo in fast_repos else f'/data1/sk/repos/{repo}_save_{c}'
        if FRESH_INTE:
            if os.path.exists(save_path) and not fresh:
                with open(save_path + '_cls_elv', 'rb') as fb:
                    st = time.time()
                    cls_elv = pkl.load(fb)
                    eprint(str(time.time()-st))
                with open(save_path, 'rb') as fb:
                    st = time.time()
                    f = pkl.load(fb)
                    eprint(str(time.time()-st))

                with open(save_path + '_mro', 'rb') as fb:
                    st = time.time()
                    mro = pkl.load(fb)
                    eprint(str(time.time()-st))
                
            else:

                
                cls_elv = load_cls_elv()
                mro = load_mro()
                if c == 'store_fast':
                    f = read_frames(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flow_{repo}.txt")
                else:    
                    f = read_frames_biend(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flow_{repo}.txt")
                with open(save_path, 'wb') as fb:
                    pkl.dump(f, fb)
                
                with open(save_path + '_mro', 'wb') as fb:
                    pkl.dump(mro, fb)
                with open(save_path + '_cls_elv', 'wb') as fb:
                    pkl.dump(cls_elv, fb)
                
                os.remove(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flow_{repo}.txt")
                os.remove(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flow_err_{repo}.txt")
                os.remove(f"/home/user/purepython/cpython-3.9/pydyna/{c}_flowc_{repo}.txt")
                os.remove(f"/home/user/purepython/cpython-3.9/pydyna/{c}_mro_{repo}.txt")
                
            frames[c] = [mro, cls_elv, f]
            class_objects_meth_sensi, global_class_classes, class_objects, class_objects_cons, class_objects_cons_para, class_objects_cons_loc, class_objects_loc_sensi, class_objects_evo_sensi, class_objects_para_sensi, class_objects_para_evo_sensi, OOMs = analyzers[c]()
        else:
            st = time.time()
            with open(f'save_inter/global_class_classes_{repo}', 'rb') as f:
                global_class_classes = pkl.load(f)
            with open(f'save_inter/class_objects_{repo}', 'rb') as f:
                class_objects = pkl.load(f)
            with open(f'save_inter/class_objects_cons_{repo}', 'rb') as f:
                class_objects_cons = pkl.load(f)
            with open(f'save_inter/class_objects_cons_para_{repo}', 'rb') as f:
                class_objects_cons_para = pkl.load(f)

            with open(f'save_inter/class_objects_cons_loc_{repo}', 'rb') as f:
                class_objects_cons_loc = pkl.load(f)
            with open(f'save_inter/class_objects_loc_sensi_{repo}', 'rb') as f:
                class_objects_loc_sensi = pkl.load(f)
            with open(f'save_inter/class_objects_evo_sensi_{repo}', 'rb') as f:
                class_objects_evo_sensi = pkl.load(f)
            with open(f'save_inter/class_objects_para_sensi_{repo}', 'rb') as f:
                class_objects_para_sensi = pkl.load(f)
            with open(f'save_inter/class_objects_para_evo_sensi_{repo}', 'rb') as f:
                class_objects_para_evo_sensi = pkl.load(f)
            with open(f'save_inter/class_objects_meth_sensi_{repo}', 'rb') as f:
                class_objects_meth_sensi = pkl.load(f)
            with open(f'save_inter/OOMs_{repo}', 'rb') as f:
                OOMs = pkl.load(f)
            with open(save_path + '_cls_elv', 'rb') as fb:
                cls_elv = pkl.load(fb)
            with open(f'save_inter/CPATTERN_DUP_{repo}', 'rb') as f:
                CPATTERN_DUP = pkl.load(f)
            with open(f'save_inter/APATTERN_DUP_{repo}', 'rb') as f:
                APATTERN_DUP = pkl.load(f)
            with open(f'save_inter/PAT2_{repo}', 'rb') as f:
                PAT2 = pkl.load(f)

            with open(f'save_inter/con_objs_{repo}', 'rb') as f:
                con_objs = pkl.load(f)

            with open(f'save_inter/class_objs_{repo}', 'rb') as f:
                class_objs = pkl.load(f)


            with open(f'save_inter/UNCOND_{repo}', 'rb') as f:
                UNCOND = pkl.load(f)
            with open(f'save_inter/ALLCOND_{repo}', 'rb') as f:
                ALLCOND = pkl.load(f)
            with open(f'save_inter/HASCOND_{repo}', 'rb') as f:
                HASCOND = pkl.load(f)
            with open(f'save_inter/OTHERCOND1_{repo}', 'rb') as f:
                OTHERCOND1 = pkl.load(f)
            with open(f'save_inter/OTHERCOND2_{repo}', 'rb') as f:
                OTHERCOND2 = pkl.load(f)
            with open(f'save_inter/OTHERCOND3_{repo}', 'rb') as f:
                OTHERCOND3 = pkl.load(f)

            eprint(str(time.time()-st))
        result_report(cls_elv, class_objects_meth_sensi, global_class_classes, class_objects, class_objects_cons, class_objects_cons_para, class_objects_cons_loc, class_objects_loc_sensi, class_objects_evo_sensi, class_objects_para_sensi, class_objects_para_evo_sensi, OOMs, con_objs)
        with open(f'result_text/q2/evolv_clas_{repo}.txt', 'w+') as f:
            f.write(json.dumps(list(OOMs['evolve'].classes)))
        

def str_list(l):
    return [str(x) for x in l]

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # 同时显示数值和占比的饼图
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

def average(data):
    return sum(data) / len(data)

def medium(data):
    return sorted(data)[int(len(data)/2)]
def statistics(data, f):
    pass
    # f.write(str(min(data)) + '\n')
    # f.write('Apathetic: ')
    # for i, d in enumerate(data):
    # if data[0] <= 0.01:
    #     f.write('Apathetic\n')
    # elif data[0] >= 0.50:
    #     f.write('Enthusiastic\n')
    # else:
    #     f.write('Normal\n')
    # f.write('Enthusiastic: ')
    # for i, d in enumerate(data):
    #     if d >= 0.99:
    #         f.write(str(i) + ' ')
    # f.write('\n')
    
    # f.write(str(max(data)) + '\n')
    # # f.write('mean: ' + str(average(data))+ '\n')
    # f.write(str(medium(data)) + '\n')
        # f.write('variance: ' + str(variance(data)) + '\n')
        
        
if GENERATE_FIG:
    # 'rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    # 'newspaper', 'impacket', 'routersploit', 'pre_commit', 
    # seaborn is actually here, but do not run normally. 
    # 
    # 
    # repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    # 'newspaper', 'impacket', 'routersploit', 'pre_commit', 'jinja', 'pendulum', 'wordcloud', 'pinyin',  'nltk', 
    # 'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair', 'snorkel', 'pyod', 'arrow']
    # repos = []
    # repos = ['torch']
    # repos = ['thefuck']
    # repos = ['jinja', 'pendulum', 'wordcloud', 'nltk', 'pinyin', 'arrow'] 
    # repos = ['markdown','typer', 'seaborn','itsdangerous', 'altair', 'dvc']
    # repos = ['pyro','stanza', 'pywhat','snorkel', 'pyod', 'statsmodels', 'featuretools', 'pypdf', 'bandit', 'isort', 'weasyprint', 'jedi', ]
    # #  
    # repos = ['mypy', 'sklearn']    
    # repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 
    # 'pelican', 'newspaper', 'impacket', 'routersploit', 'pre_commit']
    
    # repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 
    # 'faker', 'sklearn', 'pandas', 'pelican', 'newspaper', 'impacket', 'routersploit', 'pre_commit']
    
    # 
    # 
    # repos = [ 'rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'black', 'networkx', 'click', 'pydantic', 
    # 'yapf', 'faker', 'sklearn', 'pandas', 'mypy', 'pelican', 'newspaper', 'impacket', 'routersploit', 'pre_commit', 
    # 'jinja', 'pendulum', 'wordcloud', 'nltk', 'pinyin', 'arrow', 'markdown','typer', 'seaborn','itsdangerous',
    # 'altair', 'dvc', 'pyro','stanza', 'pywhat','snorkel', 'pyod', 'statsmodels', 'featuretools', 'pypdf',
    # 'bandit', 'isort', 'weasyprint', 'jedi', 'tinydb', 'prompt_toolkit', 'torch', 'kornia', 'icecream', 'sphinx'
    # ]
    # possible mro problem: stanza(stanza.models.common.doc.Span), pyro(pyro.distributions.transforms.haar.HaarTransform), pydantic(pydantic.main.ParsingModel[Dict[int, int]])
    # 

    # repos = ['markdown', 'black', 'yapf', 'faker', 'mypy', 'pandas', 'torch', 'kornia', 'pyro']
    # repos = ['sphinx', 'kornia', 'dvc', 'yapf', 'faker', 'jedi', 'sklearn', 'torch', 'pyro', 'seaborn','stanza',
    # 'weasyprint', 'mypy', 'featuretools', 'prompt_toolkit', 'statsmodels', 'pypdf', 'black', 'tinydb', 'pandas', 'markdown']
    if os.path.exists(f'result_text/ind/{repo}.txt'):
        os.remove(f'result_text/ind/{repo}.txt')
    repos = [repo]
    Repos = [x[0].upper() for x in repos]

    for r in repos:
        repo = r
        eprint(r)
        process_repo()
    # with open(f"class_tables/CA", "rb") as f:
    #     CA = pkl.load(f)
    # with open(f"class_tables/ca_matplotlib", "rb") as f:
    #     CA2 = pkl.load(f)
    #     for c in CA2 :
    #         if c not in CA:
    #             CA[c] = CA2[c]
    # with open(f"class_tables/CA", "wb") as f:
    #     pkl.dump(CA, f)

    # with open(f"class_tables/CA", "rb") as f:
    #     CA2 = pkl.load(f)
    # exit()

    with open(f'result_text/q1/{repo}_con_objs.txt', 'w+') as f:
        # for cons, t in sorted(con_objs.items(), key = lambda x: x[1]):
        f.write(json.dumps(con_objs))
    
    with open(f'result_text/q1/{repo}_class_objs.txt', 'w+') as f:
        # for cons, t in sorted(class_objs.items(), key = lambda x: x[1]):

        #     f.write(cons + '           ' + str(t) + '\n')
        f.write(json.dumps(class_objs))
    # Question 1.1-1.3
    # draw a proportion graph
    
    with open(f'result_text/q1/{repo}.txt', 'w+') as f:

        f.write(str(POLY_CONS) + '\n')
        f.write(str(POLY_CONS_HYB) + '\n')
        f.write(str(POLY_CONS_EXT) + '\n')
        f.write(str(POLY_CONS_TYP) + '\n')
        f.write(str(POLY_CONS_PARA) + '\n')
        f.write(json.dumps(POLY_CONS_LOC) + '\n')
        f.write(str(EATTR) + '\n')
        f.write(str(NATTR) + '\n')
        f.write(str(SIM_ATTR) + '\n')
        f.write(str(STRU_ATTR) + '\n')
        f.write(str(COER_ATTR) + '\n')
        f.write(str(OTHER_ATTR) + '\n')
        
        f.write(str(sum(VS1.values())) + '\n')
        
        # draw a histogram
        f.write(json.dumps(VS1) + '\n')
        
        f.write(json.dumps(DIFF) + '\n')

    # Question 1.4-1.6

    with open(f'result_text/q2/{repo}.txt', 'w+') as f:
        for name in ['extend', 'update','remove', 'evolve']:
            f.write(str(ALL_OOM[name][0]) + '\n')
            f.write(str(ALL_OOM[name][1]) + '\n')
            statistics([x[0] for x in ALL_OOMs[name]], f)
            f.write(str(ALL_OOM[name][2]) + '\n')
            f.write(str(ALL_OOM[name][3]) + '\n')
            statistics([x[1] for x in ALL_OOMs[name]], f)
        for name in ['abnormal', 'abextend', 'abmonotonic']:
            f.write(str(ALL_OOM[name][0]) + '\n')
            f.write(str(ALL_OOM[name][1]) + '\n')
            statistics([1 - x[0] for x in ALL_OOMs[name]], f)
            f.write(str(ALL_OOM[name][2]) + '\n')
            f.write(str(ALL_OOM[name][3]) + '\n')
            statistics([1 - x[1] for x in ALL_OOMs[name]], f)
    with open(f'result_text/q2/{repo}_prop.txt', 'w+') as f:
        for name in ['evolve', 'HYB', 'EXT', 'TYP']:
            f.write(str(ALL_OOM[name][2]) + '\n')
    with open(f'result_text/q2/{repo}_pie.txt', 'w+') as f:
        f.write(json.dumps({k : len(APATTERN_DUP['evolve'][k]) for k in APATTERN_DUP['evolve']}) + '\n')
        f.write(json.dumps({k : len(CPATTERN_DUP['update'][k]) for k in CPATTERN_DUP['update']}) + '\n')
    with open(f'result_text/q2/{repo}_other.txt', 'w+') as f:
        f.write(json.dumps(PAT2) + '\n')
    with open(f'result_text/q2/{repo}_cond.txt', 'w+') as f:
        f.write(str(len(UNCOND)) + '\n')
        f.write(str(len(ALLCOND)) + '\n')
        
        f.write(str(len(HASCOND)) + '\n')
        f.write(str(len(OTHERCOND1)) + '\n')
        f.write(str(len(OTHERCOND2)) + '\n')
        f.write(str(len(OTHERCOND3)) + '\n')
        
    with open(f'result_text/q2/{repo}_inlinear.txt', 'w+') as f:
        for name in ['inlinear-value']:
            f.write(str(ALL_OOM[name][0]) + '\n')
            f.write(str(ALL_OOM[name][1]) + '\n')
            f.write(str(ALL_OOM[name][2]) + '\n')
            f.write(str(ALL_OOM[name][3]) + '\n')
       
        
    # Question 1.7-1.8
    # draw a proportion graph
    with open(f'result_text/q3/{repo}.txt', 'w+') as f:
        f.write(str(FUN_CLS2) + '\n')
        f.write(str(FUN_CLS) + '\n')
        f.write(str(EXT_CLS) + '\n')
        f.write(str(MOD_CLS) + '\n')
        f.write(str(DEL_CLS) + '\n')
        for name in ['override', 'object_func']:
            f.write(str(ALL_OOM[name][0]) + '\n')
            f.write(str(ALL_OOM[name][1]) + '\n')
            statistics([x[0] for x in ALL_OOMs[name]], f)
            f.write(str(ALL_OOM[name][2]) + '\n')
            f.write(str(ALL_OOM[name][3]) + '\n')
            statistics([x[1] for x in ALL_OOMs[name]], f)
        
    with open(f'result_text/q3/{repo}_pie.txt', 'w+') as f:
        f.write(json.dumps({k : len(CPATTERN_DUP['override'][k]) for k in CPATTERN_DUP['override']}) + '\n')
    # Question 1.9
    # draw a proportion graph
    with open(f'result_text/q4/{repo}.txt', 'w+') as f:
    #     f.write(str(sum(VV1)) + '\n')
    #     f.write(str(sum(VV2)) + '\n')
    # # draw a histogram
    #     f.write(json.dumps(VS2) + '\n')
        f.write(str(POLY_CLS) + '\n')
        f.write(str(METH_POLY_CLS) + '\n')
    

    # Question 2

    with open(f'result_text/q5/{repo}_stat.txt', 'w+') as f:
        f.write(str(POLY_CONS2) + '\n')
        f.write(str(POLY_HYB) + '\n')
        f.write(str(STA_CLS) + '\n')
        
        f.write(str(POLY_CONS_HYB2) + '\n')
        f.write(str(POLY_CONS_EXT2) + '\n')
        f.write(str(POLY_CONS_TYP2) + '\n')
        f.write(str(NATTR2) + '\n')
        f.write(str(EATTR2) + '\n')
        f.write(str(NNEATTR2) + '\n')
        
        f.write(str(SIM_ATTR2) + '\n')
        f.write(str(STRU_ATTR2) + '\n')
        f.write(str(COER_ATTR2) + '\n')
        f.write(str(OTHER_ATTR2) + '\n')
        
        f.write(json.dumps(DIFF2) + '\n')
    with open(f'result_text/q5/{repo}_cons.txt', 'w+') as f:
        f.write(str(len(YES_SITE0)) + '\n')
        f.write(str(len(NOMINAL_ATTR0)) + '\n')
        f.write(str(len(UNION_SITE0)) + '\n')
        f.write(str(len(GUARDED_SITE0)) + '\n')
        f.write(str(len(REFINE_SITE0)) + '\n')
    
    with open(f'result_text/q5/{repo}_mis.txt', 'w+') as f:
        f.write(str(len(CONMIS)) + '\n')
        f.write(str(len(MIS)) + '\n')
    with open(f'result_text/q5/{repo}.txt', 'w+') as f:
        f.write(str(len(POLY_ATTR)) + '\n')
        f.write(str(len(ALL_ATTR)) + '\n')
        f.write(str(len(YES_SITE)) + '\n')
        f.write(str(len(NOMINAL_ATTR)) + '\n')
        f.write(str(len(UNION_SITE)) + '\n')
        f.write(str(len(GUARDED_SITE)) + '\n')
        f.write(str(len(GUARDED_SITE_AGGRE)) + '\n')
        
        f.write(str(len(REFINE_SITE)) + '\n')
        
        f.write(json.dumps([len(x) for x in SENSI_SITE1]) + '\n')
        f.write(str(len(SENSI_SITE2)) + '\n')
        f.write(str(len(SENSI_SITE3)) + '\n')
        f.write(str(len(SENSI_SITE4)) + '\n')
        f.write(str(len(CAUSE_ATTR0)) + '\n')
        f.write(str(len(CAUSE_ATTR1)) + '\n')
        f.write(str(len(CAUSE_ATTR2)) + '\n')
        
        f.write(str(div(len(POLY_ATTR), len(ALL_ATTR))) + '\n')
        f.write(str(div(len(NOMINAL_ATTR), len(YES_SITE))) + '\n')
        f.write(str(div(len(UNION_SITE), len(YES_SITE))) + '\n')
        f.write(str(div(len(GUARDED_SITE), len(YES_SITE))) + '\n')
        f.write(str(div(len(REFINE_SITE), len(YES_SITE))) + '\n')
        
        f.write(str(div(len(SENSI_SITE1), len(YES_SITE))) + '\n')
        f.write(json.dumps([div(len(x), len(YES_SITE)) for x in SENSI_SITE2]) + '\n')
        f.write(str(div(len(SENSI_SITE3), len(YES_SITE))) + '\n')
        f.write(str(div(len(SENSI_SITE4), len(YES_SITE))) + '\n')
        f.write(str(div(len(CAUSE_ATTR0), len(YES_SITE))) + '\n')
        f.write(str(div(len(CAUSE_ATTR1), len(YES_SITE))) + '\n')
        f.write(str(div(len(CAUSE_ATTR2), len(YES_SITE))) + '\n')
        
    with open(f'result_text/ind/{repo}.txt','w+') as f:
        pass
    exit() 
    print(GET_SET)
    print(METHODS_DESCR)
    print(CMETHODS_DESCR)
    print(WRAPPER_DESCR)

    print(GET_SET2)
    print(METHODS_DESCR2)
    print(CMETHODS_DESCR2)
    print(WRAPPER_DESCR2)
    print(SC)
    # if FRESH_ANNOTATION:
    #     with open(f'results/setattr.txt', 'w+') as f:
    #         f.write('\n'.join(DO))

        # n = 2
    # x = np.arange(len(repos) + 1)
    # total_width = 0.8
    # width = total_width / n
    # x = x - (total_width - width)

    # for name in oom_names:
    #     data = [x[0] for x in ALL_OOMs[name]] + [ALL_OOM[name][0]/ALL_OOM[name][1]]
    #     plt.bar(x, data, width = width, tick_label=Repos + ["All"], label = "OBJ")
    #     data = [x[1] for x in ALL_OOMs[name]] + [ALL_OOM[name][2]/ALL_OOM[name][3]]
    #     plt.bar(x + width, data, width = width, tick_label=Repos + ["All"], label = "CLS")
    #     plt.legend()
    #     plt.savefig(f"results/ooms/{name}.png", dpi=300)
    #     plt.clf()
    #     statistics([x[0] for x in ALL_OOMs[name]], f"results/ooms/{name}_statistic_obj.txt")
    #     statistics([x[1] for x in ALL_OOMs[name]], f"results/ooms/{name}_statistic_cls.txt")
    
    # x = np.arange(len(repos) + 1)
    # total_width = 0.8
    # width = total_width / n
    # x = x - (total_width - width)
    # for name in oom_names:
    #     data = [1-x[0] for x in ALL_OOMs[name]] + [1- ALL_OOM[name][0]/ALL_OOM[name][1]]
    #     plt.bar(x, data, width = width, tick_label=Repos + ["All"], label = "OBJ")
    #     data = [1-x[1] for x in ALL_OOMs[name]] + [1- ALL_OOM[name][2]/ALL_OOM[name][3]]
    #     plt.bar(x + width, data, width = width, tick_label=Repos + ["All"], label = "CLS")
    #     plt.legend()
    #     plt.savefig(f"results/ooms/anti-{name}.png", dpi=300)
    #     plt.clf()
    with open('results/mutate_t1t2.txt', 'w+') as f:

        f.write('$t_1=none$ & ' + ' & '.join(M1C) + '\\\\ \\hline' +'\n')
        f.write('$t_2=none$ & ' + ' & '.join(M2C) + '\\\\ \\hline' +'\n')
        f.write('$t_1=_n t_2$ & ' + ' & '.join(M3C)+ '\\\\ \\hline' +'\n')
        f.write('$t_1 <:_n t_2$ & ' + ' & '.join(M4C) + '\\\\ \\hline' +'\n')
        f.write('$t_2 <:_n t_1$ & ' + ' & '.join(M5C) + '\\\\ \\hline' +'\n')
        f.write('$t1 \sim_n t_2$ & ' + ' & '.join(M6C) + '\\\\ \\hline' +'\n')
        f.write('$others$ & ' + ' & '.join(M7C) + '\\\\ \\hline' +'\n')

    with open('results/overload_t1t2.txt', 'w+') as f:

        f.write('$t_1=none$ & ' + ' & '.join(O1C) + '\\\\ \\hline' +'\n')
        f.write('$t_2=none$ & ' + ' & '.join(O2C) + '\\\\ \\hline' +'\n')
        f.write('$t_1=_n t_2$ & ' + ' & '.join(O3C)+ '\\\\ \\hline' +'\n')
        f.write('$t_1 <:_n t_2$ & ' + ' & '.join(O4C) + '\\\\ \\hline' +'\n')
        f.write('$t_2 <:_n t_1$ & ' + ' & '.join(O5C) + '\\\\ \\hline' +'\n')
        f.write('$t1 \sim_n t_2$ & ' + ' & '.join(O6C) + '\\\\ \\hline' +'\n')
        f.write('$others$ & ' + ' & '.join(O7C) + '\\\\ \\hline' +'\n')
    with open('results/mutate_overload.txt', 'w+') as f:

        f.write('mutate & ' + ' & '.join(SAM) + '\\\\ \\hline' +'\n')
        f.write('overload & ' + ' & '.join(SAO) + '\\\\ \\hline' +'\n')
    
    

    print('\n'.join(S))
    if not ONLY_CLASS_ANALYSIS:
        # with open('results/object_table1.txt', 'w+') as f:

        #     f.write('Extension & ' + ' & '.join(E) + '\\\\ \\hline' +'\n')
        #     f.write('Deletion & ' + ' & '.join(D) + '\\\\ \\hline' +'\n')

        with open('results/operations_and_reflections.txt', 'w+') as f:
            for key in FUN:
                f.write(f'(): {key} & ' + ' & '.join(str_list(FUN[key])) + '\\\\ \\hline' +'\n')
            for key in DOP:
                f.write(f'[]: {key} & ' + ' & '.join(str_list(DOP[key])) + '\\\\ \\hline' +'\n')
            
        with open('results/set/ext_events.txt', 'w+') as f:

            f.write('$extenion\ set$ & ' + ' & '.join(EXT_SET) + '\\\\ \\hline' +'\n')
            f.write('$extenion\ event$ & ' + ' & '.join(EXT_EVENT) + '\\\\ \\hline' +'\n')

        n = 2
        x = np.arange(3)
        
        total_width = 0.8
        width = total_width / n
        x = x - (total_width - width)
        
            
        # plt.legend()
        # plt.savefig(f"results/ooms/overall.png")
        # plt.clf()

        n = 2
        x = np.arange(len(BUILTIN_DESCRIPTORS))
        
        total_width = 0.8
        width = total_width / n
        x = x - (total_width - width)
        NN = []
        data1 = []
        data2 = []
        for descr in BUILTIN_DESCRIPTORS:
            if SDALL[descr][0] > 0:
                NN.append(descr.split('.')[-1])
                x = SDALL[descr]
                # data1.append(x[0]/x[1])
                data1.append(x[0])
                
                # data2.append(x[2]/x[3])
                data2.append(x[2])
                
        # NN = [x.split('.')[-1] for x in BUILTIN_DESCRIPTORS]
        # data = [x[0]/x[1] for x in SDALL.values()]
        # plt.bar(x, data, width = width, tick_label=NN, label = "OBJ")
        plt.pie(data1, labels = NN, autopct=make_autopct(data1))
        plt.savefig(f"results/set/set-descr/overall.png", dpi=300)
        plt.cla()
        # data = [x[2]/x[3] for x in SDALL.values()]
        # plt.bar(x + width, data, width = width, tick_label=NN, label = "CLS")
        # plt.bar(np.arange(len(BUILTIN_DESCRIPTORS)), data2)
        plt.pie(data2, labels = NN, autopct=make_autopct(data2))
        # plt.legend()
        plt.savefig(f"results/set/set-descr/overall_dup.png", dpi=300)
        plt.cla()

        n = 2
        x = np.arange(len(BUILTIN_DESCRIPTORS))
        
        total_width = 0.8
        width = total_width / n
        x = x - (total_width - width)
        NN = []
        data1 = []
        data2 = []
        for descr in BUILTIN_DESCRIPTORS:
            if LDALL[descr][0] > 0:
                NN.append(descr.split('.')[-1])
                x = LDALL[descr]
                # data1.append(x[0]/x[1])
                data1.append(x[0])
                
                # data2.append(x[2]/x[3])
                data2.append(x[2])
        # NN = [x.split('.')[-1] for x in BUILTIN_DESCRIPTORS]
        # data = [x[0]/x[1] for x in SDALL.values()]
        # plt.bar(x, data, width = width, tick_label=NN, label = "OBJ")
        plt.pie(data1, labels = NN, autopct=make_autopct(data1))
        plt.savefig(f"results/get/get-descr/overall.png", dpi=300)
        plt.cla()
        # data = [x[2]/x[3] for x in SDALL.values()]
        # plt.bar(x + width, data, width = width, tick_label=NN, label = "CLS")
        plt.pie(data2, labels = NN, autopct=make_autopct(data2))
        # plt.legend()
        plt.savefig(f"results/get/get-descr/overall_dup.png", dpi=300)
        plt.cla()


        NN = [x[0].upper() for x in OM]
        data = [x[0]/x[1] for x in ALL_OOM.values()]
        # plt.bar(x, data, width = width, tick_label=NN, label = "OBJ")
        with open(f"results/ooms/overall_obj.txt", 'w+') as f:
            f.write(str(OM)+'\n')
            f.write(str(data)+'\n')
            f.write('\n')
        data = [x[2]/x[3] for x in ALL_OOM.values()]
        # plt.bar(x + width, data, width = width, tick_label=NN, label = "CLS")
        with open(f"results/ooms/overall_cls.txt", 'w+') as f:
            f.write(str(OM)+'\n')
            f.write(str(data)+'\n')


        
        SSSS = SSS1 + SSS2 + SSS3 + SSS4 + SSS5 + SSS6 + SSS7
        # SSSL = [SSS1/SSSS, SSS2/SSSS,  SSS6/SSSS, SSS3/SSSS,  SSS7/SSSS, SSS4/SSSS, SSS5/SSSS,]
        SSSL = [SSS1, SSS2,  SSS6, SSS3,  SSS7, SSS4, SSS5,]
        SSSN = ['set-descr','update', 'modification','override',  'overstep', 'extension', 'initialization', ]
        plt.cla()
        plt.pie(SSSL, labels = SSSN, autopct=make_autopct(SSSL))
        # plt.legend()
        plt.savefig("results/set/RoE.png", dpi=300)
        with open("results/set/Events.txt", "w+") as f:
            f.write(str(SSSS))
        

        SSLS = SSL1 + SSL2 + SSL3 + SSL4 + SSL5 + SSL6 + SSL7
        # SSLL = [SSL1/SSLS, SSL2/SSLS,  SSL6/SSLS, SSL3/SSLS, SSL7/SSLS, SSL4/SSLS, SSL5/SSLS, ]
        SSLL = [SSL1, SSL2,  SSL6, SSL3, SSL7, SSL4, SSL5, ]
        # SSLL = [SSL1, SSL2,  SSL6, SSL3, SSL7, SSL4, SSL5/SSLS, ]
        SSSN = ['set-descr','update',  'modification', 'override', 'overstep', 'extension', 'initialization', ]
        plt.clf()
        # plt.bar(np.arange(len(SSLL)), SSLL, tick_label = SSSN)
        plt.pie(SSLL, labels = SSSN, autopct=make_autopct(SSLL))
        # plt.legend()
        plt.savefig("results/set/RoL.png", dpi=300)
        with open("results/set/Locations.txt", "w+") as f:
            f.write(str(SSLS))
        
        with open('results/set/update/upd.txt', 'w+') as f:
            f.write(f'{UPD}: {NON_UPD}: {UPD/(UPD + NON_UPD)}')
        with open('results/set/override/ovr.txt', 'w+') as f:
            f.write(f'{OVR}: {NON_OVR}: {OVR/(OVR + NON_OVR)}')
        
        
        with open('results/set/set-descr/descriptors_set.txt', 'w+') as f:
            for d in SD:
                d_ = d.replace('_', '\_')
                f.write(f'${d_}$ & ' + ' & '.join(SD[d]) + '\\\\ \\hline' +'\n')

        with open('results/set/update/object_t1t2.txt', 'w+') as f:

            f.write('$t_1=none$ & ' + ' & '.join(M1[:10]) + '\\\\ \\hline' +'\n')
            f.write('$t_2=none$ & ' + ' & '.join(M2[:10]) + '\\\\ \\hline' +'\n')
            # f.write('$t_1=_n t_2$ & ' + ' & '.join(M3)+ '\\\\ \\hline' +'\n')
            f.write('$t_1 <:_n t_2$ & ' + ' & '.join(M4[:10]) + '\\\\ \\hline' +'\n')
            f.write('$t_2 <:_n t_1$ & ' + ' & '.join(M5[:10]) + '\\\\ \\hline' +'\n')
            f.write('$t1 <:_s t_2$ & ' + ' & '.join(M8[:10]) + '\\\\ \\hline' +'\n')
            f.write('$t1 \sim_n t_2$ & ' + ' & '.join(M6[:10]) + '\\\\ \\hline' +'\n')
            
            f.write('$others$ & ' + ' & '.join(M7[:10]) + '\\\\ \\hline' +'\n')

            f.write('$t_1=none$ & ' + ' & '.join(M1[10:]) + '\\\\ \\hline' +'\n')
            f.write('$t_2=none$ & ' + ' & '.join(M2[10:]) + '\\\\ \\hline' +'\n')
            # f.write('$t_1=_n t_2$ & ' + ' & '.join(M3)+ '\\\\ \\hline' +'\n')
            f.write('$t_1 <:_n t_2$ & ' + ' & '.join(M4[10:]) + '\\\\ \\hline' +'\n')
            f.write('$t_2 <:_n t_1$ & ' + ' & '.join(M5[10:]) + '\\\\ \\hline' +'\n')
            f.write('$t1 <:_s t_2$ & ' + ' & '.join(M8[10:]) + '\\\\ \\hline' +'\n')
            f.write('$t1 \sim_n t_2$ & ' + ' & '.join(M6[10:]) + '\\\\ \\hline' +'\n')
            
            f.write('$others$ & ' + ' & '.join(M7[10:]) + '\\\\ \\hline' +'\n')
        
        with open('results/set/override/object_t1t2.txt', 'w+') as f:

            f.write('$t_1=none$ & ' + ' & '.join(M1ov[:10]) + '\\\\ \\hline' +'\n')
            f.write('$t_2=none$ & ' + ' & '.join(M2ov[:10]) + '\\\\ \\hline' +'\n')
            # f.write('$t_1=_n t_2$ & ' + ' & '.join(M3ov)+ '\\\\ \\hline' +'\n')
            f.write('$t_1 <:_n t_2$ & ' + ' & '.join(M4ov[:10]) + '\\\\ \\hline' +'\n')
            f.write('$t_2 <:_n t_1$ & ' + ' & '.join(M5ov[:10]) + '\\\\ \\hline' +'\n')
            f.write('$t1 <:_s t_2$ & ' + ' & '.join(M8ov[:10]) + '\\\\ \\hline' +'\n')
            f.write('$t1 \sim_n t_2$ & ' + ' & '.join(M6ov[:10]) + '\\\\ \\hline' +'\n')
            
            
            f.write('$others$ & ' + ' & '.join(M7ov[:10]) + '\\\\ \\hline' +'\n')


            f.write('$t_1=none$ & ' + ' & '.join(M1ov[10:]) + '\\\\ \\hline' +'\n')
            f.write('$t_2=none$ & ' + ' & '.join(M2ov[10:]) + '\\\\ \\hline' +'\n')
            # f.write('$t_1=_n t_2$ & ' + ' & '.join(M3ov)+ '\\\\ \\hline' +'\n')
            f.write('$t_1 <:_n t_2$ & ' + ' & '.join(M4ov[10:]) + '\\\\ \\hline' +'\n')
            f.write('$t_2 <:_n t_1$ & ' + ' & '.join(M5ov[10:]) + '\\\\ \\hline' +'\n')
            f.write('$t1 <:_s t_2$ & ' + ' & '.join(M8ov[10:]) + '\\\\ \\hline' +'\n')
            f.write('$t1 \sim_n t_2$ & ' + ' & '.join(M6ov[10:]) + '\\\\ \\hline' +'\n')
            
            
            f.write('$others$ & ' + ' & '.join(M7ov[10:]) + '\\\\ \\hline' +'\n')
        kv = CPATTERN['override'].items()
        print(kv)
        PTS = [x[1] for x in kv]
        LBS = [x[0] for x in kv]
        plt.cla()
        plt.pie(PTS, labels = LBS, autopct=make_autopct(PTS))
        # plt.legend()
        plt.savefig("results/set/override/distribution_t1t2.png", dpi=300)
        with open("results/set/override/distribution_t1t2.txt", "w+") as f:
            f.write(str(PTS))
                

        kv = CPATTERN_DUP['override'].items()
        print(kv)
        PTS = [len(x[1]) for x in kv]
        print(PTS)
        LBS = [x[0] for x in kv]
        plt.cla()
        plt.pie(PTS, labels = LBS, autopct=make_autopct(PTS))
        # plt.legend()
        plt.savefig("results/set/override/distribution_t1t2_dup.png", dpi=300)
        with open("results/set/override/distribution_t1t2_dup.txt", "w+") as f:
            f.write(str(PTS))
        
        kv = CPATTERN['update'].items()
        print(kv)
        PTS = [x[1] for x in kv]
        LBS = [x[0] for x in kv]
        plt.cla()
        plt.pie(PTS, labels = LBS, autopct=make_autopct(PTS))
        # plt.legend()
        plt.savefig("results/set/update/distribution_t1t2.png", dpi=300)
        with open("results/set/update/distribution_t1t2.txt", "w+") as f:
            f.write(str(PTS))

        kv = CPATTERN_DUP['update'].items()
        print(kv)
        PTS = [len(x[1]) for x in kv]
        print(PTS)
        LBS = [x[0] for x in kv]
        plt.cla()
        plt.pie(PTS, labels = LBS, autopct=make_autopct(PTS))
        # plt.legend()
        plt.savefig("results/set/update/distribution_t1t2_dup.png", dpi=300)
        with open("results/set/update/distribution_t1t2_dup.txt", "w+") as f:
            f.write(str(PTS))
        
        kv = APATTERN['update'].items()
        print(kv)
        PTS = [x[1] for x in kv]
        LBS = [x[0] for x in kv]
        plt.cla()
        plt.pie(PTS, labels = LBS, autopct=make_autopct(PTS))
        # plt.legend()
        plt.savefig("results/set/update/distribution_alias.png", dpi=300)
        with open("results/set/update/distribution_alias.txt", "w+") as f:
            f.write(str(PTS))
        

        kv = APATTERN_DUP['update'].items()
        print(kv)
        PTS = [len(x[1]) for x in kv]
        print(PTS)
        LBS = [x[0] for x in kv]
        plt.cla()
        plt.pie(PTS, labels = LBS, autopct=make_autopct(PTS))
        # plt.legend()
        plt.savefig("results/set/update/distribution_alias_dup.png", dpi=300)
        with open("results/set/update/distribution_alias_dup.txt", "w+") as f:
            f.write(str(PTS))

        kv = APATTERN['extend'].items()
        print(kv)
        PTS = [x[1] for x in kv]
        LBS = [x[0] for x in kv]
        plt.cla()
        plt.pie(PTS, labels = LBS, autopct=make_autopct(PTS))
        # plt.legend()
        plt.savefig("results/set/extend/distribution_alias.png", dpi=300)
        with open("results/set/extend/distribution_alias.txt", "w+") as f:
            f.write(str(PTS))

        kv = APATTERN_DUP['extend'].items()
        print(kv)
        PTS = [len(x[1]) for x in kv]
        print(PTS)
        LBS = [x[0] for x in kv]
        plt.cla()
        plt.pie(PTS, labels = LBS, autopct=make_autopct(PTS))
        # plt.legend()
        plt.savefig("results/set/extend/distribution_alias_dup.png", dpi=300)
        with open("results/set/extend/distribution_alias_dup.txt", "w+") as f:
            f.write(str(PTS))

        LSSS = LSS1 + LSS2 + LSS3
        # LSSL = [LSS1/LSSS, LSS2/LSSS, LSS3/LSSS]
        LSSL = [LSS1, LSS2, LSS3]
        LSSN = ['get-descr','inheritance','lookup']
        plt.cla()
        plt.pie(LSSL, labels = LSSN, autopct=make_autopct(LSSL))
        # plt.legend()
        plt.savefig("results/get/distribution.png", dpi=300)
        with open("results/get/distribution.txt", "w+") as f:
            f.write(str(LSSS))

        LSLS = LSL1 + LSL2 + LSL3
        # LSLL = [LSL1/LSLS, LSL2/LSLS, LSL3/LSLS]
        LSLL = [LSL1, LSL2, LSL3]
        LSSN = ['get-descr','inheritance','lookup']
        plt.cla()
        plt.pie(LSLL, labels = LSSN, autopct=make_autopct(LSLL))
        # plt.legend()
        plt.savefig("results/get/distribution_dup.png", dpi=300)
        with open("results/get/distribution_dup.txt", "w+") as f:
            f.write(str(LSLS))
        
        with open('results/get/get-descr/descriptors_get.txt', 'w+') as f:
            for d in LD:
                d_ = d.replace('_', '\_')
                f.write(f'${d_}$ & ' + ' & '.join(LD[d]) + '\\\\ \\hline' +'\n')

        depth_to_print = []
        depth_kv_to_print = []
        for r in repos:
            with open(f'results/get/inherit/depth_{r}', 'w+') as f:
                f.write('average: ' + str(div(sum(DEPTH_ATTR_REPO[r]), len(DEPTH_ATTR_REPO[r]))) + '\n')
                f.write('times: ' + str(len(DEPTH_ATTR_REPO[r])))
                depth_to_print.append(div(sum(DEPTH_ATTR_REPO[r]), len(DEPTH_ATTR_REPO[r])))
            # with open(f'results/get/inherit/depth-kv_{r}', 'w+') as f:
            #     object_nonobject = [0, 0]
            #     for k, v in sorted(inherited_repo[r].items(), key=lambda x: x[1]):
            #         f.write(str(k) + '   ' + str(v) + '\n')
            #         if k in OBJ_ATTRS:
            #             object_nonobject[0] += v
            #         else:
            #             object_nonobject[1] += v
            #     f.write(str(div(object_nonobject[0], (object_nonobject[0] + object_nonobject[1]))))
            #     depth_kv_to_print.append(div(object_nonobject[0], (object_nonobject[0] + object_nonobject[1])))
        statistics([x for x in depth_to_print if x != -1], 'results/get/inherit/depth_stat')
        # statistics([x for x in depth_kv_to_print if x != -1], 'results/get/inherit/depth-kv_stat')

        n = 1
        x = np.arange(3)
        plt.clf()
        total_width = 0.8
        width = total_width / n
        x = x - (total_width - width)
        # NN = [x[0].upper() for x in OM]
        AL = sum(x[1] for x in DEPTH_ATTR_LEN.items())
        data = [DEPTH_ATTR_LEN[1]/AL, DEPTH_ATTR_LEN[2]/AL, sum(x[1] for x in DEPTH_ATTR_LEN.items() if x[0]>2)/AL]
        plt.bar(x, data, width = width, tick_label=["1", "2", ">2"], label = "depth")
        plt.legend()
        plt.savefig(f"results/get/inherit/depth_len.png", dpi=300)
        plt.clf()

        total_width = 0.8
        width = total_width / n
        x = x - (total_width - width)
        # NN = [x[0].upper() for x in OM]
        AL = sum(x[1] for x in DEPTH_DESCR_LEN.items())
        data = [DEPTH_DESCR_LEN[1]/AL, DEPTH_DESCR_LEN[2]/AL, sum(x[1] for x in DEPTH_DESCR_LEN.items() if x[0]>2)/AL]
        plt.bar(x, data, width = width, tick_label=["1", "2", ">2"], label = "depth")
        plt.legend()
        plt.savefig(f"results/get/get-descr/depth_len.png", dpi=300)
        plt.clf()

        with open('results/get/inherit/depth', 'w+') as f:
            f.write('average: ' + str(sum(DEPTH_ATTR)/len(DEPTH_ATTR)) + '\n')
            f.write('times: ' + str(len(DEPTH_ATTR)))
        with open('results/get/inherit/depth-kv', 'w+') as f:
            # object_nonobject = [0, 0]
            # for k, v in sorted(inherited.items(), key=lambda x: x[1]):
            #     f.write(str(k) + '   ' + str(v) + '\n')
            #     if k in OBJ_ATTRS:
            #         object_nonobject[0] += v
            #     else:
            #         object_nonobject[1] += v
            f.write(str(div(inherited[0], (inherited[0] + inherited[1]))))
        with open('results/get/get-descr/depth', 'w+') as f:
            f.write('average: ' + str(sum(DEPTH_DESCR)/len(DEPTH_DESCR))+'\n')
            f.write('times: ' + str(len(DEPTH_DESCR)))
        with open('results/get/get-descr/depth-kv', 'w+') as f:
            # object_nonobject = [0, 0]
            # for k, v in sorted(inherited_descr.items(), key=lambda x: x[1]):
            #     f.write(str(k) + '   ' + str(v) + '\n')
            #     if k in OBJ_ATTRS:
            #         object_nonobject[0] += v
            #     else:
            #         object_nonobject[1] += v
            f.write(str(div(inherited_descr[0], (inherited_descr[0] + inherited_descr[1]))))

        DSSS = DSS1 + DSS2
        # DSSL = [DSS1/DSSS, DSS2/DSSS]
        DSSL = [DSS1, DSS2]
        DSSN = ['del-descr','remove']
        plt.cla()
        plt.pie(DSSL, labels = DSSN, autopct=make_autopct(DSSL))
        # plt.legend()
        plt.savefig("results/del/distribution.png", dpi=300)
        with open("results/del/distribution.txt", "w+") as f:
            f.write(str(DSSS))
        
        DSLS = DSL1 + DSL2
        # DSLL = [DSL1/DSLS, DSL2/DSLS]
        DSLL = [DSL1, DSL2]
        DSLN = ['del-descr','remove']
        plt.cla()
        plt.pie(DSLL, labels = DSLN, autopct=make_autopct(DSLL))
        # plt.legend()
        plt.savefig("results/del/distribution_dup.png", dpi=300)

        with open("results/del/distribution_dup.txt", "w+") as f:
            f.write(str(DSLS))


        with open('results/global/object_variation.txt', 'w+') as f:

            f.write('$n = 1$ & ' + ' & '.join(V1) + '\\\\ \\hline' +'\n')
            f.write('$n = 2$ & ' + ' & '.join(V2) + '\\\\ \\hline' +'\n')
            f.write('$n = 3$ & ' + ' & '.join(V3)+ '\\\\ \\hline' +'\n')
            f.write('$n > 3$ & ' + ' & '.join(V4) + '\\\\ \\hline' +'\n')
        
        with open('results/others/object_syntaxes.txt', 'w+') as f:

            f.write('. & ' + ' & '.join(I1) + '\\\\ \\hline' +'\n')
            f.write('() & ' + ' & '.join(I3)+ '\\\\ \\hline' +'\n')
            f.write('[] & ' + ' & '.join(I2) + '\\\\ \\hline' +'\n')

            f.write('? & ' + ' & '.join(I0) + '\\\\ \\hline' +'\n')

        
        with open('results/set/dis_mod.txt', 'w+') as f:

            f.write('explicit & ' + ' & '.join(EXP1[:10]) + '\\\\ \\hline' +'\n')
            f.write('method & ' + ' & '.join(IMP_MEM1[:10]) + '\\\\ \\hline' +'\n')
            f.write('function & ' + ' & '.join(IMP_FUN1[:10]) + '\\\\ \\hline' +'\n')
            f.write('alias & ' + ' & '.join(IMP_ALI1[:10]) + '\\\\ \\hline' +'\n')
            
            f.write('explicit & ' + ' & '.join(EXP1[10:]) + '\\\\ \\hline' +'\n')
            f.write('method & ' + ' & '.join(IMP_MEM1[10:]) + '\\\\ \\hline' +'\n')
            f.write('function & ' + ' & '.join(IMP_FUN1[10:]) + '\\\\ \\hline' +'\n')
            f.write('alias & ' + ' & '.join(IMP_ALI1[10:]) + '\\\\ \\hline' +'\n')
            
            # f.write('implicit alias & ' + ' & '.join(IMP_ALI1) + '\\\\ \\hline' +'\n')
        
        with open('results/set/dis_ext.txt', 'w+') as f:

            f.write('explicit & ' + ' & '.join(EXP2[:10]) + '\\\\ \\hline' +'\n')
            f.write('method & ' + ' & '.join(IMP_MEM2[:10]) + '\\\\ \\hline' +'\n')
            f.write('function & ' + ' & '.join(IMP_FUN2[:10]) + '\\\\ \\hline' +'\n')
            f.write('alias & ' + ' & '.join(IMP_ALI2[:10]) + '\\\\ \\hline' +'\n')
            
            f.write('explicit & ' + ' & '.join(EXP2[10:]) + '\\\\ \\hline' +'\n')
            f.write('method & ' + ' & '.join(IMP_MEM2[10:]) + '\\\\ \\hline' +'\n')
            f.write('function & ' + ' & '.join(IMP_FUN2[10:]) + '\\\\ \\hline' +'\n')
            f.write('alias & ' + ' & '.join(IMP_ALI2[10:]) + '\\\\ \\hline' +'\n')
            
            # f.write('implicit alias & ' + ' & '.join(IMP_ALI2) + '\\\\ \\hline' +'\n')

        with open('results/others/abnormal.txt', 'w+') as f:

            f.write('d1:=d2 & ' + ' & '.join(R) + '\\\\ \\hline' +'\n')
            f.write('d.clear() & ' + ' & '.join(C) + '\\\\ \\hline' +'\n')
            f.write('d1.union(d2) & ' + ' & '.join(U) + '\\\\ \\hline' +'\n')

            f.write('reflection & ' + ' & '.join(REFL) + '\\\\ \\hline' +'\n')
            f.write(str(F) + '\n')
            f.write(str(F2) + '\n')

        # plt.cla()
        # n = 2
        # x = np.arange(len(repos))
        
        # total_width = 0.8
        # width = total_width / n
        # x = x - (total_width - width)
        # plt.bar(x + width, P, width = width, tick_label=Repos, label = "POLY")
        # plt.legend()
        # plt.savefig("results/global/object_polymorphism.png")

        # plt.cla()
        # n = 2
        # x = np.arange(len(repos))
        
        # total_width = 0.8
        # width = total_width / n
        # x = x - (total_width - width)
        # plt.bar(x, APLOYMORPHISM[0], width = width, tick_label=Repos, label = "OBJ")
        # plt.bar(x + width, OAPLOYMORPHISM[0], width = width, tick_label=Repos, label = "CLS")
        # plt.legend()
        # plt.savefig("results/global/object_polymorphism_pot.png")


        # plt.cla()
        # n = 2
        # x = np.arange(len(repos))
        
        # total_width = 0.8
        # width = total_width / n
        # x = x - (total_width - width)
        # plt.bar(x, APLOYMORPHISM[1], width = width, tick_label=Repos, label = "OBJ")
        # plt.bar(x + width, OAPLOYMORPHISM[1], width = width, tick_label=Repos, label = "CLS")
        # plt.legend()
        # plt.savefig("results/global/object_polymorphism.png")

        # print(CPLOYMORPHISM)
        print([APLOYMORPHISM[1], APLOYMORPHISM[2], sum(APLOYMORPHISM[x] for x in APLOYMORPHISM if x >= 3)])
        print([OAPLOYMORPHISM[1], OAPLOYMORPHISM[2], sum(OAPLOYMORPHISM[x] for x in OAPLOYMORPHISM if x >= 3)])
        print(J1)
        print(J1_)
        print(J2)
        print(J2_)

        print(len(J1s))
        print(len(J1_s))
        print(len(J2s))
        print(len(J2_s))
        print(APLOYMORPHISM_POT)
        print(OAPLOYMORPHISM_POT)
        # statistics(APLOYMORPHISM[0], "results/global/class_pot_polymorphism.txt")
        # statistics(APLOYMORPHISM[1], "results/global/class_pos_polymorphism.txt")
        # statistics(OAPLOYMORPHISM[0], "results/global/object_pot_polymorphism.txt")
        # statistics(OAPLOYMORPHISM[1], "results/global/object_pos_polymorphism.txt")
        
        d = dict(sorted(PAT.items(), key=lambda item: len(item[1]), reverse=True))
        for i, item in enumerate(d.items()):
            print(item)
        d = dict(sorted(PAT2.items(), key=lambda item: item[1], reverse=True))
        for i, item in enumerate(d.items()):
            print(item)
        
        print(OTHERS_SET)
        print(OTHERS_GET)
        print('rns')
        print(RNS)
        print('lns')
        print(LNS)
        print(GETSET)
        print('\n'.join(DELETE_INFOS))
        print(NNUM)
        print(SNUM)
        # with open('results/set/SL.txt', 'w+') as f:

        #     f.write('$descriptors$ & ' + ' & '.join(LL1) + '\\\\ \\hline' +'\n')
        #     f.write('$class attributes$ & ' + ' & '.join(LL2) + '\\\\ \\hline' +'\n')
        #     f.write('$object attributes$ & ' + ' & '.join(LL3)+ '\\\\ \\hline' +'\n')
        #     f.write('$descriptors$ & ' + ' & '.join(SL1) + '\\\\ \\hline' +'\n')
        #     f.write('$object attributes$ & ' + ' & '.join(SL2) + '\\\\ \\hline' +'\n')
        #     f.write('$n = 3$ & ' + ' & '.join(SL3)+ '\\\\ \\hline' +'\n')
        
else:
    process_repo()


        # n = 2
        # x = np.arange(len(repos))
        
        # total_width = 0.8
        # width = total_width / n
        # x = x - (total_width - width)
        # data = [x[0] for x in M]
        # plt.bar(x, data, width = width, tick_label=Repos, label = "OBJ")
        # data = [x[1] for x in M]
        # plt.bar(x + width, data, width = width, tick_label=Repos, label = "CLS")
        # plt.legend()
        # plt.savefig("results/set/object_modification.png")
        # plt.clf()

        # n = 2
        # x = np.arange(len(repos))
        
        # total_width = 0.8
        # width = total_width / n
        # x = x - (total_width - width)
        # data = [x[0] for x in E]
        # plt.bar(x, data, width = width, tick_label=Repos, label = "OBJ")
        # data = [x[1] for x in E]
        # plt.bar(x + width, data, width = width, tick_label=Repos, label = "CLS")
        # plt.legend()
        # plt.savefig("results/set/object_extension.png")
        # plt.clf()
