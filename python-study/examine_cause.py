from collections import defaultdict
import pickle as pkl
from random import sample

# attrs = set()
# with open('used_loc', 'r') as f:
#     for line in f.readlines():
#         attrs.add(line.strip())

# attrs2 = set()
# with open(f'choosed.txt', 'r') as f:
#     lines = f.readlines()
#     for i, line in enumerate(lines):
#         if line.startswith('-------case--------'):
#             attr = line.strip().split('-------case--------')[1]
#             attrs2.add(attr)
#             types = lines[i+1]
#             locs = []
#             for line2 in lines[i+2:]:
#                 if line2 == '\n':
#                     break
                    
# for attr in attrs2:
#     if attr not in attrs:
#         print(attr)
# 
all_repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    'newspaper', 'impacket', 'routersploit', 'pre_commit',  'pendulum', 'wordcloud', 'pinyin',  
    'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair',  'nltk', 'jinja',  'pyod', 'arrow', 
    'kornia','stanza', 'jedi',  'faker', 'black','weasyprint',
    'yapf', 'pyro',  'featuretools',  'dvc', 'torch', 'mypy', 
    'pypdf', 'pandas', 'seaborn','statsmodels',  'snorkel', 
    'prompt_toolkit', 'markdown' ,'spacy', 'sklearn', 'sphinx',
    ]
# all_repos = [ 'pywhat', 'icecream',  'pendulum', 'pre_commit', 'faker'] # 4
# all_repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
#     'newspaper', 'impacket', 'routersploit', 'pre_commit', 'jinja', 'pendulum', 'wordcloud', 'pinyin',  'nltk', 
#     'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair','pyod', 'arrow'] # 28
class OOM:
    def __init__(self) -> None:
        self.t = False
        self.all = 0
        self.tot = 0
        self.classes = set()
cnt = 0
repo_cnt = defaultdict(int)
cls = set()
all_cls = 0
L = set()
cc = 0
bases = {}
attr_faults = defaultdict(list)

for repo in all_repos:
    with open(f'manual/cause_to_examine_manual/{repo}.txt', 'r') as fr:
        lines = fr.readlines()
        try:
            line = lines[-1]
            cnt += int(line.split(":")[-1])
        except Exception :
            pass

print(cnt)


for repo in all_repos:
    with open(f'result_text/ind/{repo}.txt','r') as f:
        pass

    with open(f'save_inter/OOMs_{repo}', 'rb') as f:
        OOMs = pkl.load(f)
    
    with open(f'cause_to_examine/ext_{repo}.txt', 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            loc = lines[i]
            att = lines[i+1]
            cls.add('.'.join(att.split('.')[:-1]))
            L.add(loc)
            # cnt += 1
    all_cls += OOMs['extend2'].all
    # cnt += len(OOMs['extend2'].classes)
    # with open(f'cause_to_examine/{repo}.txt', 'r') as fr:
    #     lines = fr.readlines()
    #     for i in range(0, len(lines), 4):
    #         attr = lines[i]
    #         tys = lines[i + 1]
    #         loc = lines[i + 2]
    #         cont = lines[i + 3]
    #         cnt += 1
    #         repo_cnt[repo] += 1
    #         attr_faults[attr].append((tys, loc))

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

    # with open('cause_to_examine/conditionals.txt', 'a+') as f:
    #     for loc in OTHERCOND3:
    #         f.write(loc + '\n')
        # print(OTHERCOND3)

    with open(f'class_to_examine/{repo}.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('-------case--------'):
                attr = line.strip().split('-------case--------')[1]
                types = lines[i+1]
                
                locs = []
                for line2 in lines[i+2:]:
                    if line2 == '\n':
                        break
                    locs.append(line2)
                assert attr not in bases
                if len(locs) == 0:
                    cc += 1
                else:
                    bases[attr] = (types, locs)
                    if 'missing' in types:
                        cc += 0
# choosed = sample(bases.keys(), 1)
# with open('choosed2.txt', 'a+') as f:
#     for c in choosed:
#         if c not in attrs:
#             f.write(c+'\n')
#             f.write(bases[c][0])
#             for loc in bases[c][1]:
#                 f.write(loc)
print(cc)

# print(len(attr_faults))
# # print(all_cls)
# print(len(cls))
# choosed = sample(attr_faults.keys(), 100)
# choosed_mis = 0
# # with open('choosed_for_mypy.txt', 'w+') as f: 
# #     for c in choosed:
# #         f.write('-------case--------' + c)
# #         for ts, loc in attr_faults[c]:
# #             f.write(ts)
# #             f.write(loc)
# #             if 'missing' in ts:
# #                 choosed_mis += 1
# #             break
# #         f.write('\n')
# # print(choosed_mis)
# bases2 = defaultdict(list)
# def decode_types(s):
#     r = s.strip()[1:-1].split(',')
#     return {rx.strip()[1:-1] for rx in r}
# # with open(f'choosed.txt', 'r') as f:
# #     lines = f.readlines()
# #     for i, line in enumerate(lines):
# #         if line.startswith('-------case--------'):
# #             attr = line.strip().split('-------case--------')[1]
# #             types = lines[i+1]
            
# #             locs = []
# #             for line2 in lines[i+2:]:
# #                 # if line2 == '\n':
# #                 #     break
# #                 # if '|' in line2:
# #                 #     loc = line2.split('|')[0].strip()
# #                 #     a = line2.split('|')[1].strip()
# #                 #     ts = decode_types(a)
# #                 #     bases2[loc].append((attr, ts))
# #                 # else:
# #                 #     loc = line2.split('|')[0].strip()
# #                 #     ts = decode_types(types)
# #                 #     if 'missing' in ts:
# #                 #         ts.remove("missing")
# #                 #     bases2[loc].append((attr, ts))
# #                 locs.append(line2)
# #             assert attr not in bases2
# #             if len(locs) == 0:
# #                 cc += 1
# #             else:
# #                 bases2[attr] = (types, locs)
# #                 if len(bases2) == 100:
# #                     print(attr)
# #                 if 'missing' in types:
# #                     cc += 0
# # print(cc)
# # cnt = 0
