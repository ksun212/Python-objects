import pickle as pkl
import json
repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    'newspaper', 'impacket', 'routersploit', 'pre_commit',  'pendulum', 'wordcloud', 'pinyin',  
    'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair',  'nltk', 'jinja',  'pyod', 'arrow', 
    'kornia','stanza', 'jedi',  'faker', 'sklearn','black',
    'yapf', 'pyro',  'featuretools',  'dvc', 'torch', 'mypy', 'weasyprint',
    'pypdf', 'pandas', 'sphinx','seaborn','statsmodels',  'snorkel', 
    'prompt_toolkit', 'markdown' ,'spacy', 
    ]

repos = ['seaborn']
# no_repos = ['statsmodels', 'sklearn', 'pypdf', 'jedi', 'black', 'mypy', 'torch', 'kornia']

for repo in repos:
    with open(f"/data/repos/{repo}_save_store_attr", 'rb') as fb, open(f"/data/repos/{repo}_save_store_attr2", 'w+') as fw:
        f = pkl.load(fb)
        for name in f:
            # for global_id, info in f[name]:
            fw.write(json.dumps((name, f[name])) + '\n')

    # with open(f'/data/repos/{repo}_save_store_attr2') as f:
    #     print(repo)
    #     for line in f:
    #         obj, infos = json.loads(line)
    #         break
    #         for global_id, info in infos:
    #             pass