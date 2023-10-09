from collections import defaultdict
import pickle as pkl
# 
all_repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    'newspaper', 'impacket', 'routersploit', 'pre_commit',  'pendulum', 'wordcloud', 'pinyin',  
    'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair',  'nltk', 'jinja',  'pyod', 'arrow', 
    'kornia','stanza', 'jedi',  'faker', 'black',
    'yapf', 'pyro',  'featuretools',  'dvc', 'torch', 'mypy', 'weasyprint',
    'pypdf', 'pandas', 'seaborn','statsmodels',  'snorkel', 
    'prompt_toolkit', 'markdown' ,'spacy', 'sphinx','sklearn',
    ]

all_repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    'newspaper', 'impacket', 'routersploit', 'pre_commit', 'jinja', 'pendulum', 'wordcloud', 'pinyin',  'nltk', 
    'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair','pyod', 'arrow'] # 28
cnt = 0
repo_cnt = defaultdict(int)
for repo in all_repos:
    with open(f'cause_to_examine/{repo}.txt', 'r') as fr:
        lines = fr.readlines()
        for i in range(0, len(lines), 4):
            attr = lines[i]
            tys = lines[i + 1]
            loc = lines[i + 2]
            cont = lines[i + 3]
            cnt += 1
            repo_cnt[repo] += 1

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
    a = 1


#     print(OTHERCOND3)
print(cnt)
# cnt = 0
# for repo in all_repos:
#     with open(f'cause_to_examine copy 2/{repo}.txt', 'r') as fr:
#         lines = fr.readlines()
#         try:
#             line = lines[-1]
#             cnt += int(line.split(":")[-1])
#         except Exception :
#             pass
# print(cnt)