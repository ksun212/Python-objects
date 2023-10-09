repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    'newspaper', 'impacket', 'routersploit', 'pre_commit', 'jinja', 'pendulum', 'wordcloud', 'pinyin',  'nltk', 
    'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair','pyod', 'arrow'] # 28


# repos = ['faker', 'kornia','jedi',]

# repos = ['black', 'sklearn','stanza', ] # 4
# repos = ['yapf', 'pyro',  'featuretools',  'dvc',  'torch',] # 11
# repos = ['pypdf', 'pandas', 'sphinx','seaborn','statsmodels'] # 16
# repos = ['prompt_toolkit', 'markdown',  'weasyprint' ,'spacy', 'snorkel', 'mypy',] # 22
# repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
#     'newspaper', 'impacket', 'routersploit', 'pre_commit',  'pendulum', 'wordcloud', 'pinyin',  
#     'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair',  'nltk', 'jinja',  'pyod', 'arrow', 
#     'kornia','stanza', 'jedi',  'faker', 'sklearn','black',
#     'yapf', 'pyro',  'featuretools',  'dvc', 'torch', 'mypy', 'weasyprint',
#     'pypdf', 'pandas', 'sphinx','seaborn','statsmodels',  'snorkel', 
#     'prompt_toolkit', 'markdown' ,'spacy', 
#     ]
# repos = ['mypy']
import os
for repo in repos:
    # os.system(f"nohup python /home/user/thefuck-master/pre_run_biend.py {repo} 1 0 0 > /home/user/thefuck-master/outs/{repo}_out 2>&1 &")
    os.system(f"nohup python /home/user/thefuck-master/pre_run_biend.py {repo} 1 0 1 > /home/user/thefuck-master/outs/{repo}_out 2>&1 &")
    # os.system(f"nohup python /home/user/thefuck-master/pre_run_biend.py {repo} 1 1 1 > /home/user/thefuck-master/outs/{repo}_out1 2>&1 &")

# class Absent:
#     pass   
# class DecisionTree:
#     def __init__(self):
#         self.num = 42
#         self.n_classes = None
#     def init(self) -> None:
#         pass
#     def fit(self, x:int):
#         self.n_classes = x
#     def predict(self):
#         return self.n_classes > 0
        
#     def ten(self):
#         return 10
    
# tree = DecisionTree() # D0
# tree.init() # D1
# tree.fit() # D2
# tree.predict() # D3



# a1 = A(1)
# a2 = A(None)
# a2.x = 42
# print(a1.x + 1)

