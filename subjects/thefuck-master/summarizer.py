from collections import defaultdict
import json
# 'sklearn',
q5 = False
LOC_K = 4
ABS_LOC = False
# networkx, pinyin,  sklearn, nltk, altair, pyod, kornia, stanza, featuretools, dvc, torch, pandas, seaborn, statsmodels, spacy, snorkel
# pydantic, typer, bandit, isort, arrow, jedi, black, yapf, mypy
# requests, flask, impacket, routersploit, itsdangerous, pelican, sphinx
# rich, thefuck, cookiecutter, click, prompt_toolkit
# jinja, pypdf, markdown, weasyprint
# pywhat, icecream,  pendulum, pre_commit, faker
# newspaper, wordcloud, pyro, pyecharts
if q5:
    sci_repo = ['networkx', 'pinyin', 'nltk', 'altair', 'pyod', 'kornia','stanza', 'featuretools', 'dvc', 'pandas', 'seaborn', 'statsmodels', 'spacy', 'snorkel', 'torch'] # 16
    sklearn = ['sklearn']
else:
    sci_repo = ['networkx', 'pinyin',  'sklearn', 'nltk', 'altair', 'pyod', 'kornia','stanza', 'featuretools', 'dvc', 'torch', 'pandas', 'seaborn', 'statsmodels', 'spacy', 'snorkel'] # 16
web_repo = ['requests', 'flask', 'impacket', 'routersploit', 'itsdangerous', 'pelican',  'sphinx'] # 7
prog_tool_repo = ['pydantic','typer', 'bandit', 'isort', 'arrow', 'jedi', 'black', 'yapf', 'mypy'] # 10
cli_tool = ['rich', 'thefuck', 'cookiecutter', 'click', 'prompt_toolkit'] # 5
format_tool = ['jinja', 'pypdf', 'markdown', 'weasyprint'] # 4
prog_utility = [ 'pywhat', 'icecream',  'pendulum', 'pre_commit', 'faker'] # 4
others = ['newspaper', 'wordcloud', 'pyro','pyecharts'] # 4
# 
# 
if q5:
    #  
    all_repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    'newspaper', 'impacket', 'routersploit', 'pre_commit',  'pendulum', 'wordcloud', 'pinyin',  
    'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair',  'jinja',  'pyod', 'arrow', 
    'kornia','stanza', 'faker',  'nltk',  'black','pandas','jedi',
    'yapf', 'pyro',  'featuretools',  'dvc', 'torch', 'jinja', 
    'pypdf', 'sphinx','seaborn','statsmodels',  'snorkel', 'mypy',
    'prompt_toolkit', 'markdown',  'weasyprint' ,'spacy', 
    ]
    # all_repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    # 'newspaper', 'impacket', 'routersploit', 'pre_commit',  'pendulum', 'wordcloud', 'pinyin',  
    # 'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair',  'nltk', 'jinja',  'pyod', 'arrow', 
    # 'kornia','stanza', 'jedi', 'black', 'faker',
    # 'yapf', 'pyro',  'featuretools',  'dvc',  
    # 'pypdf', 'pandas', 'seaborn','statsmodels', 
    # 'prompt_toolkit', 'markdown',  'weasyprint' ,'spacy',
    # ]
else:
    #  
    all_repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
    'newspaper', 'impacket', 'routersploit', 'pre_commit',  'pendulum', 'wordcloud', 'pinyin',  
    'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair',  'nltk', 'jinja',  'pyod', 'arrow', 
    'kornia','stanza', 'jedi',  'faker', 'sklearn', 'black',
    'yapf', 'pyro',  'featuretools',  'dvc', 'torch', 'mypy', 'weasyprint',
    'pypdf', 'pandas', 'sphinx','seaborn','statsmodels',  'snorkel', 
    'prompt_toolkit', 'markdown' ,'spacy', 
    ]

# 'black',
# all_repos = ['thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
#     'newspaper', 'impacket', 'routersploit', 'pre_commit',  'pendulum', 'wordcloud', 'pinyin',  
#     'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair',  'nltk', 'jinja',  'pyod', 'arrow', 
#     'kornia','stanza', 'jedi',  'faker', 'sklearn', 
#     'yapf', 'pyro',  'featuretools',  'dvc', 'torch', 'mypy', 'weasyprint',
#     'pypdf', 'pandas', 'sphinx','seaborn','statsmodels',  'snorkel', 
#     'prompt_toolkit', 'markdown' ,'spacy', 
#     ]
# # 
# all_repos = ['rich', 'thefuck', 'requests', 'flask', 'cookiecutter', 'pyecharts', 'networkx', 'click', 'pydantic', 'pelican', 
#     'newspaper', 'impacket', 'routersploit', 'pre_commit',  'pendulum', 'wordcloud', 'pinyin',  
#     'typer', 'itsdangerous', 'pywhat', 'bandit', 'isort', 'icecream',  'altair',  'jinja', 'pyod', 'arrow', ]

# all_repos = ['rich']



def medium(data):
    return sorted(data)[int(len(data)/2)]
def average(data):
    # return sum(data) / len(data)
    return medium(data)

def statistics(data, f):
    f.write(str(min(data)) + '\n')
    f.write(str(max(data)) + '\n')
    f.write(str(medium(data)) + '\n')

def read_tuple(tup_str):
    s = tup_str[1:-1].split(',')
    s = [int(x.strip()) for x in s]
    return s

def add_list(l1, l2):
    assert len(l1) == len(l2)
    for i in range(len(l1)):
        l1[i] += l2[i]
def add_dict(d1, d2):
    for k in d2:
        if k not in d1:
            d1[k] = d2[k]
        else:
            if isinstance(d2[k], list):
                d1[k].extend(d2[k])
            d1[k] += d2[k]
def sci_notm(x):
    if not isinstance(x, int):
        x = float(x)
    if x == -1:
        return 'nan'
    
    x = 1-x
    return "%.2f" % x

def sci_not(x):
    if not isinstance(x, int):
        x = float(x)
    if x == -1:
        return '*'
    return "%.2f" % x
if q5:
    NAME = {1: 'SCI', 2: 'PROG', 3: 'WEB', 4: 'CLI', 5: 'FORMAT', 6: 'UTILITY', 7: 'OTHERS', 8: 'SKLEARN', 9:'ALL - SKLEARN'}
    OJB_CNT = ['$3\\times 10^6$', '$1\\times 10^6$', '$2\\times 10^5$', '$5\\times 10^4$', '$1\\times 10^6$', '$1\\times 10^5$', '$1\\times 10^5$', '$3\\times 10^5$', '$6\\times 10^6$']
else:
    NAME = {1: 'SCI', 2: 'PRG', 3: 'WEB', 4: 'TER', 5: 'FMT', 6: 'UTL', 7: 'OTH', 8: 'ALL'}
    OJB_CNT = ['$3\\times 10^6$', '$1\\times 10^6$', '$2\\times 10^5$', '$5\\times 10^4$', '$1\\times 10^6$', '$1\\times 10^5$', '$3\\times 10^5$', '$6\\times 10^6$']
with open(f'result_text/q1f/cons_project_wise.txt', 'w+') as f1, open(f'result_text/q1f/obj_project_wise.txt', 'w+') as f2, open(f'result_text/q1f/overall_project_wise1.txt', 'w+') as f4, open(f'result_text/q2f/project_wise2.txt', 'w+') as f5:
    
    
    if q5:
        R = [sci_repo, prog_tool_repo, web_repo, cli_tool, format_tool, prog_utility, others, sklearn, all_repos]
    else:
        R = [sci_repo, prog_tool_repo, web_repo, cli_tool, format_tool, prog_utility, others, all_repos]
    for now, repos in enumerate(R):
    # for now, repos in enumerate([all_repos]):
        POLY_CONS = 0
        POLY_CONS_HYB = 0
        POLY_CONS_EXT = 0
        POLY_CONS_TYP = 0
        POLY_CONS_PARA = 0
        POLY_CONS_LOC = [0 for i in range(LOC_K)]
        ALL_CONS = 0
        NATTR = 0
        EATTR = 0
        SIM_ATTR = 0

        STRU_ATTR = 0
        COER_ATTR = 0 
        OTHER_ATTR = 0

        UNCOND = 0 
        HASCOND = 0
        OTHERCOND1 = 0
        OTHERCOND2 = 0
        OTHERCOND3 = 0
        
        ALLCOND = 0
        VS1 = {}
        VS3 = [0, 0, 0]
        CLS_NUM = {}
        POLY_CONSS = []


        for repo in repos:

            with open(f'result_text/ind/{repo}.txt','r') as f:
                pass
            with open(f'result_text/q1/{repo}.txt', 'r') as f:
                lines = f.readlines()
                POLY_CONS += int(lines[0].strip())
                
                POLY_CONS_HYB += int(lines[1].strip())
                POLY_CONS_EXT += int(lines[2].strip())
                POLY_CONS_TYP += int(lines[3].strip())
                POLY_CONS_PARA += int(lines[4].strip())
                add_list(POLY_CONS_LOC, json.loads(lines[5].strip()))
                EATTR += int(lines[6].strip())
                NATTR += int(lines[7].strip())
                SIM_ATTR += int(lines[8].strip())
                STRU_ATTR += int(lines[9].strip())
                COER_ATTR += int(lines[10].strip())
                OTHER_ATTR += int(lines[11].strip())
                
                ALL_CONS += int(lines[12].strip())
                POLY_CONSS.append(int(lines[0].strip())/int(lines[12].strip()))
                add_dict(VS1, json.loads(lines[13].strip()))
                add_list(VS3, json.loads(lines[14].strip()))
        # Figure 3
        f1.write(NAME[now+1] + ' & ' + str(int(ALL_CONS)) + ' & ' + sci_not(POLY_CONS/ALL_CONS) + ' & ' + sci_not(average(POLY_CONSS)) + ' \\\\ ' + '\n')
        
        VS1 = sorted(VS1.items(), key = lambda x: int(x[0]))
        with open(f'result_text/q1f/point.txt', 'w+') as f:
            for k, v in VS1:
                f.write(k+'\t' + str(v) + '\n')

        # Figure 4
        with open(f'result_text/q3f/para.txt', 'w+') as f:
            f.write(str(POLY_CONS-POLY_CONS_PARA) + '\t' + str(POLY_CONS_PARA)+ '\n')
            f.write(str(POLY_CONS-POLY_CONS_LOC[3]) + '\t' + str(POLY_CONS_LOC[3]) + '\n')
            f.write(str(POLY_CONS-POLY_CONS_LOC[2]) + '\t' + str(POLY_CONS_LOC[2]) + '\n')
            f.write(str(POLY_CONS-POLY_CONS_LOC[1]) + '\t' + str(POLY_CONS_LOC[1]) + '\n')
            # f.write(str(POLY_CONS) + '\t' + str(POLY_CONS-POLY_CONS_PARA) + '\n')
        
        with open(f'result_text/q3f/overall.txt', 'w+') as f:
            f.write('TYPE' + '\t' + str(POLY_CONS_TYP) + '\n')
            f.write('ATTR' + '\t' + str(POLY_CONS_EXT) + '\n')
            f.write('HYB' + '\t' + str(POLY_CONS_HYB) + '\n')

        with open(f'result_text/q1f/type_relation.txt', 'w+') as f:
            f.write( str(EATTR) + '\n')
            f.write( str(NATTR) + '\n')
            f.write( str(SIM_ATTR) + '\n')
            f.write( str(COER_ATTR) + '\n')
            f.write( str(STRU_ATTR) + '\n')
            f.write( str(OTHER_ATTR) + '\n')
            
        

        oom_names = ['extend', 'override', 'update', 'remove', 'abnormal', 'abmonotonic', 'abextend', 'evolve', 'object_func', 'only-e', 'only-u', 'both','inlinear-value']
        ALL_OOMs = {k:[[], []] for k in oom_names}
        ALL_OOM = {k:[0,0,0,0] for k in oom_names}
        evolv_clas = []
        EVOS = 0
        EXTS = 0
        TYPS = 0
        HYBS = 0
        ATTI = {k:[0,0,0] for k in oom_names}
        PAT2 = {}

        APATTERN = defaultdict(int)
        CPATTERN = {}
        LL = []
        class_objs = {}
        con_objs = [{} for i in range(LOC_K)]
        q2_records = []
        for repo in repos:
            with open(f'result_text/q2/{repo}.txt', 'r') as f:
                lines = f.readlines()
                evo_obj = 0
                evo_cls = 0
                for i, name in enumerate(['extend', 'update', 'remove', 'evolve', 'abnormal', 'abextend', 'abmonotonic']):
                    ALL_OOM[name][0] += float(lines[4*i])
                    ALL_OOM[name][1] += float(lines[4*i + 1])
                    if name == 'evolve':
                        ALL_OOMs[name][0].append(float(lines[4*i])/float(lines[4*i + 1]))
                        evo_obj = float(lines[4*i])
                    ALL_OOM[name][2] += float(lines[4*i + 2])
                    ALL_OOM[name][3] += float(lines[4*i + 3])
                    if name == 'evolve':
                        ALL_OOMs[name][1].append(float(lines[4*i + 2])/float(lines[4*i + 3]))
                        evo_cls = float(lines[4*i + 2])
                    CLS_NUM[repo] = float(lines[4*i + 3])
                    # lf = 0.01 if name in ['extend', 'update', 'remove'] else 0.10
                    # rf = 0.50 if name in ['extend', 'update', 'remove'] else 0.80
                    
                    # if float(lines[8*i + 6]) <= lf:
                    #     ATTI[name][0] += 1
                    # elif float(lines[8*i + 6]) >= rf:
                    #     ATTI[name][1] += 1
                    # else:
                    #     ATTI[name][2] += 1
                    q2_records.append(lines)
                for i, name in enumerate(['extend', 'update', 'remove', 'evolve', 'abnormal', 'abextend', 'abmonotonic']):
                    if name != 'evolve':
                        if evo_obj == 0:
                            ALL_OOMs[name][0].append(0)
                        else:
                            ALL_OOMs[name][0].append(float(lines[4*i])/evo_obj)
                        if name == 'abmonotonic':
                            ALL_OOMs[name][0][-1] = 1-ALL_OOMs[name][0][-1]
                    if name != 'evolve':
                        if evo_cls == 0:
                            ALL_OOMs[name][1].append(0)
                        else:
                            ALL_OOMs[name][1].append(float(lines[4*i + 2])/evo_cls)
                        if name == 'abmonotonic':
                            ALL_OOMs[name][1][-1] = 1-ALL_OOMs[name][1][-1]
            with open(f'result_text/q2/{repo}_pie.txt', 'r') as f:
                lines = f.readlines()
                add_dict(APATTERN, json.loads(lines[0].strip()))
                add_dict(CPATTERN, json.loads(lines[1].strip()))
            with open(f'result_text/q2/{repo}_prop.txt', 'r') as f:
                lines = f.readlines()
                EVOS += int(lines[0].strip())
                HYBS += int(lines[1].strip())
                EXTS += int(lines[2].strip())
                TYPS += int(lines[3].strip())

            with open(f'result_text/q2/{repo}_other.txt', 'r') as f:
                lines = f.readlines()
                add_dict(PAT2, json.loads(lines[0].strip()))
            with open(f'result_text/q2/{repo}_cond.txt', 'r') as f:
                lines = f.readlines()
                UNCOND += int(lines[0].strip())
                ALLCOND += int(lines[1].strip())
                HASCOND += int(lines[2].strip())
                OTHERCOND1 += int(lines[3].strip())
                OTHERCOND2 += int(lines[4].strip())
                OTHERCOND3 += int(lines[5].strip())
                
                LL.append(lines)
            if ABS_LOC:
                with open(f'result_text/q2/{repo}_inlinear.txt', 'r+') as f:
                    lines = f.readlines()
                    for i, name in enumerate(['inlinear-value']):
                        ALL_OOM[name][0] += float(lines[4*i])
                        ALL_OOM[name][1] += float(lines[4*i + 1])
                        ALL_OOM[name][2] += float(lines[4*i + 2])
                        ALL_OOM[name][3] += float(lines[4*i + 3])

                with open(f'result_text/q1/{repo}_con_objs.txt', 'r+') as f:
                    lines = f.readlines()
                    for i in range(LOC_K):
                        test_dic = json.loads(lines[0].strip())[i]
                        for test_ in test_dic:
                            con_objs[i][test_] = test_dic[test_]

                
                with open(f'result_text/q1/{repo}_class_objs.txt', 'r+') as f:
                    lines = f.readlines()
                    test_dic = json.loads(lines[0].strip())
                    for test_ in test_dic:
                        class_objs[test_] = test_dic[test_]
                    # add_dict(class_objs, json.loads(lines[0].strip()))
                with open(f'result_text/q2/evolv_clas_{repo}.txt', 'r+') as f:
                    lines = f.readlines()
                    evolv_clas.extend(json.loads(lines[0].strip()))

        for name in ['evolve']:
            f2.write(NAME[now+1] + ' & ' + OJB_CNT[now] + ' & ' + str(sci_not(ALL_OOM[name][0]/ALL_OOM[name][1])) + ' & ' + str(sci_not(average(ALL_OOMs[name][0]))) + ' & ' + str(int(ALL_OOM[name][3])) + ' & ' + str(sci_not(ALL_OOM[name][2]/ALL_OOM[name][3])) + ' & ' + str(sci_not(average(ALL_OOMs[name][1])))  + ' \\\\ ' + '\n')
        
        # with open(f'result_text/q2f/hist_bev_obj.txt', 'w+') as f:
        #     for i, name in enumerate(['only-e', 'only-u', 'both', 'remove']):
        #         f.write(str(sci_not(ALL_OOM[name][0]/ALL_OOM['evolve'][0])) + '\t' + str(sci_not(ALL_OOM[name][2]/ALL_OOM['evolve'][2])) + '\n') 
        #     for i, name in enumerate(['abmonotonic']):
        #         f.write(str(sci_not(1 - ALL_OOM['abmonotonic'][0]/ALL_OOM['evolve'][0])) + '\t' + str(sci_not(1 - ALL_OOM['abmonotonic'][2]/ALL_OOM['evolve'][2])) + '\n') 
        
        if ABS_LOC:
            assert len(evolv_clas) == ALL_OOM['evolve'][2]
            multi_clas = [set(), set(), set(), set()]
            not_recency_clas = [set(), set(), set(), set()]
            multi_clas1 = [set(), set(), set(), set()]
            not_recency_clas1 = [set(), set(), set(), set()]
            cnt = [0,0,0, 0]
            all_cnt = set()
            con_objs_no_part = [defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)]
            for i in range(LOC_K):
                con_objs[i] = {k:v for k,v in con_objs[i].items() if v != 0}
                for test_ in con_objs[i]:
                    for cons in con_objs[i][test_]:
                        add_dict(con_objs_no_part[i][cons], con_objs[i][test_][cons])
                        cla = cons.split('|')[0]
                        if len(con_objs[i][test_][cons]) > 1:
                            cnt[i] += 1
                            multi_clas[i].add(cla)
                            infos = sorted(con_objs[i][test_][cons].items(), key = lambda x: int(x[1][0]))
                            cons_times = [x[1][0] for x in infos]
                            for j, (obj, (cons_time, info_)) in enumerate(infos):
                                # the timestamp of this event should be smaller than the timestamp of the next construction 
                                for global_id, info in info_:
                                    if j < len(cons_times) -1 :
                                        if global_id >= cons_times[j+1]:
                                            not_recency_clas[i].add(cla)
                        all_cnt.add(cla)
            # 563602, 564917
            for i in range(LOC_K):
                for cons in con_objs_no_part[i]:
                    cla = cons.split('|')[0]
                    if len(con_objs_no_part[i][cons]) > 1:
                        multi_clas1[i].add(cla)
                        infos = sorted(con_objs_no_part[i][cons].items(), key = lambda x: int(x[1][0]))
                        cons_times = [x[1][0] for x in infos]
                        for j, (obj, (cons_time, info_)) in enumerate(infos):
                            for global_id, info in info_:
                                if j < len(cons_times) -1 :
                                    if global_id >= cons_times[j+1]:
                                        not_recency_clas1[i].add(cla)
                
            for i in range(4):
                multi_clas[i] = multi_clas[i].intersection(set(evolv_clas))
                not_recency_clas[i] = not_recency_clas[i].intersection(set(evolv_clas))
                multi_clas1[i] = multi_clas1[i].intersection(set(evolv_clas))
                not_recency_clas1[i] = not_recency_clas1[i].intersection(set(evolv_clas))
            
            with open(f'result_text/q3f/abs_loc', 'w+') as f:
                for i in range(LOC_K):
                    # f.write(str(len(not_recency_clas[LOC_K-i-1])) + '\t' + str(len(multi_clas[LOC_K-i-1])-len(not_recency_clas[LOC_K-i-1])) + '\t' + str(len(evolv_clas)-len(multi_clas[LOC_K-i-1])) + '\n')
                    t1 = str(sci_not(len(multi_clas1[i])/len(evolv_clas)))
                    t2 = str(sci_not(len(not_recency_clas1[i])/len(evolv_clas)))
                    t3 = str(sci_not(len(multi_clas[i])/len(evolv_clas)))
                    t4 = str(sci_not(len(not_recency_clas[i])/len(evolv_clas)))
                    f.write(t1 + '\t' + t2 + '\t' + t3 + '\t' + t4 + '\n')
                    # f.write(t1 + '\t' + t2 + '\n')
                    

        Func_Name = {'function':'Function', 'method':'Method', 'non_local':'Others', 'local':'Local'}
        with open(f'result_text/q1f/obj_func.txt', 'w+') as f:
            for k in ['local', 'method', 'non_local']:
                if k == 'non_local':
                    f.write(Func_Name[k] + '\t' + str(APATTERN[k] + APATTERN['function']) + '\n')
                else:
                    f.write(Func_Name[k] + '\t' + str(APATTERN[k]) + '\n')

        with open(f'result_text/q3f/pie_cond.txt', 'w+') as f:
            f.write(str(UNCOND)+'\n')
            f.write(str(ALLCOND)+'\n')
            # f.write(str(HASCOND+OTHERCOND1+OTHERCOND2+OTHERCOND3)+'\n')
            f.write(str(OTHERCOND2)+'\n')
            f.write(str(HASCOND+OTHERCOND3)+'\n')
            # f.write(str(OTHERCOND3)+'\n')
        
        # with open(f'result_text/q2f/pie_type.txt', 'w+') as f:
        #     for k in CPATTERN:
        #         f.write(k + '\t' + str(CPATTERN[k]) + '\n')

        
        # with open(f'result_text/q2/hist_bev_cls_{now}.txt', 'w+') as f:
        #     for i, name in enumerate(['only-e', 'only-u', 'both', 'remove', 'evolve']):
        #         f.write(str(sci_not(ALL_OOM[name][2]/ALL_OOM['evolve'][3])) + '\t' + str(sci_not(average(ALL_OOMs[name][1]))) + '\n') 

        # with open(f'result_text/q2/hist_bev_att_{now}.txt', 'w+') as f:
        #     for i, name in enumerate(['only-e', 'only-u', 'both', 'remove', 'evolve']):
        #         f.write(str(ATTI[name][0]) + '\t' + str(ATTI[name][1]) + '\t' + str(ATTI[name][2]) + '\n') 
        # # with open(f'result_text/q2f/hist_pat_obj.txt', 'w+') as f:
        # #     for i, name in enumerate(['abnormal', 'abextend', 'abmonotonic']):
        # #         f.write(str(sci_not(1 - ALL_OOM[name][0]/ALL_OOM[name][1])) + '\t' + str(sci_not(1 - ALL_OOM[name][2]/ALL_OOM[name][3]))+ '\n') 
        name_map = {'only-e': 'Only-E', 'only-u':'Only-M', 'both':'E-M', 'remove': 'Del'}
        name_map = {'extend': 'Ext', 'update':'Mod', 'remove': 'Del'}
        
        with open(f'result_text/q1f/actions.txt', 'w+') as f:
            for i, name in enumerate(['extend', 'update', 'remove', ]):
                f.write(name_map[name] + ' \t '+ str(sci_not(ALL_OOM[name][0]/ALL_OOM['evolve'][0])) + ' \t ' + str(sci_not(average(ALL_OOMs[name][0]))) + ' \t ' + str(sci_not(ALL_OOM[name][2]/ALL_OOM['evolve'][2])) + ' \t ' + str(sci_not(average(ALL_OOMs[name][1]))) + '\\\\ \n' ) 
        
        with open(f'result_text/q3f/monotonic.txt', 'w+') as f:
            f.write('Mono \t ' + str(sci_not(1 - ALL_OOM['abmonotonic'][0]/ALL_OOM['evolve'][0])) + ' \t ' + str(sci_not(average(ALL_OOMs['abmonotonic'][0]))) + ' \t ' + str(sci_not(1 - ALL_OOM['abmonotonic'][2]/ALL_OOM['evolve'][2]))+ ' \t ' + str(sci_not(average(ALL_OOMs['abmonotonic'][1]))) + '\\\\ \n') 
        

        # with open(f'result_text/q2/hist_pat_cls_{now}.txt', 'w+') as f:
        #     for i, name in enumerate(['abnormal', 'abextend', 'abmonotonic']):
        #         f.write(str(sci_not(1 - ALL_OOM[name][2]/ALL_OOM[name][3])) + '\t' + str(sci_not(medium(ALL_OOMs[name][1]))) + '\n') 

        # with open(f'result_text/q2/hist_pat_att_{now}.txt', 'w+') as f:
        #     for i, name in enumerate(['abnormal', 'abextend', 'abmonotonic']):
        #         f.write(str(ATTI[name][0]) + '\t' + str(ATTI[name][1]) + '\t' + str(ATTI[name][2]) + '\n') 
                        

        EXT_CLS = 0
        EVO_CLS = 0
        DEL_CLS = 0
        FUN_CLS = 0
        FUN_CLS2 = 0
        CPATTERN = {}
        for repo in repos:
            with open(f'result_text/q3/{repo}.txt', 'r') as f:
                lines = f.readlines()
                FUN_CLS2 += int(lines[0])
                FUN_CLS += int(lines[1])
                EXT_CLS += int(lines[2])
                EVO_CLS += int(lines[3])
                DEL_CLS += int(lines[4])
                for i, name in enumerate(['override', 'object_func']):
                    ALL_OOM[name][0] += float(lines[5 + 4*i])
                    ALL_OOM[name][1] += float(lines[5 + 4*i + 1])
                    ALL_OOMs[name][0].append(float(lines[5 + 4*i])/float(lines[5 + 4*i + 1]))
                    ALL_OOM[name][2] += float(lines[5 + 4*i + 2])
                    ALL_OOM[name][3] += float(lines[5 + 4*i + 3])
                    ALL_OOMs[name][1].append(float(lines[5 + 4*i + 2])/float(lines[5 + 4*i + 3]))
            with open(f'result_text/q3/{repo}_pie.txt', 'r') as f:
                lines = f.readlines()
                add_dict(CPATTERN, json.loads(lines[0].strip()))
        with open(f'result_text/q3/pie_type_{now}.txt', 'w+') as f:
            for k in CPATTERN:
                f.write(k + '\t' + str(CPATTERN[k]) + '\n')
        # f3.write(NAME[now+1] + ' & ' + str(int(ALL_OOM['override'][3])) + ' & ' + sci_not(EXT_CLS/ALL_OOM['override'][3]) + ' & ' + sci_not(EVO_CLS/ALL_OOM['override'][3]) + ' & ' + sci_not(DEL_CLS/ALL_OOM['override'][3]) + ' & ' + sci_not(FUN_CLS/ALL_OOM['override'][3]) + ' \\\\ ' + '\n')
        # f3.write('Extension' + ' & ' + str(EXT_CLS) + ' & ' + sci_not(EXT_CLS/ALL_OOM['override'][3]) + '\\\\' + '\n')
        # f3.write('Modification' + ' & ' + str(EVO_CLS) + ' & ' + sci_not(EVO_CLS/ALL_OOM['override'][3]) + '\\\\' + '\n')
        # f3.write('Deletion' + ' & ' + str(DEL_CLS) + ' & ' + sci_not(DEL_CLS/ALL_OOM['override'][3]) + '\\\\' + '\n')
        # f3.write('Method' + ' & ' + str(FUN_CLS) + ' & ' + sci_not(FUN_CLS/ALL_OOM['override'][3]) + '\\\\' + '\n')
        

        with open(f'result_text/q3/hist_bev_obj_{now}.txt', 'w+') as f:
            for i, name in enumerate(['override']):
                f.write(str(sci_not(ALL_OOM[name][0]/ALL_OOM[name][1])) + '\t' + str(sci_not(ALL_OOM[name][2]/ALL_OOM[name][3])) + '\n') 

        with open(f'result_text/q3/hist_bev_att_{now}.txt', 'w+') as f:
            for i, name in enumerate(['override']):
                f.write(str(ATTI[name][0]) + '\t' + str(ATTI[name][1]) + '\t' + str(ATTI[name][2]) + '\n') 
        VV1 = 0 
        VV2 = 0
        VS2 = {}

        POLY_CLS = 0
        METH_POLY_CLS = 0
        for repo in repos:
            with open(f'result_text/q4/{repo}.txt', 'r') as f:
                lines = f.readlines()
                POLY_CLS += int(lines[0].strip())
                METH_POLY_CLS += int(lines[1].strip())
                # add_dict(VS2, json.loads(lines[2].strip()))
        VS2 = sorted(VS2.items(), key = lambda x: int(x[0]))
        with open(f'result_text/q4/point_{now}.txt', 'w+') as f:
            for k, v in VS2:
                f.write(k+'\t' + str(v) + '\n')

        POLY_ATTR = ALL_ATTR = YES_SITE = NOMINAL_ATTR = GUARDED_SITE = GUARDED_SITE_AGGRE = SENSI_SITE = SENSI_SITE2 = CAUSE_ATTR0 = CAUSE_ATTR1 = CAUSE_ATTR2 = UNION_SITE = 0
        # POLY_PRO, NOM_PRO, GUA_PRO, SEN_PRO, SEN2_PRO, CAU_PRO = ([] for i in range(6))
        SENSI_SITE1 = [0 for i in range(LOC_K)]
        SENSI_SITE2 = 0
        SENSI_SITE3 = 0 
        SENSI_SITE4 = 0 
        REFINE_SITE = 0 

        POLY_CONS2 = 0
        POLY_HYB = 0
        STA_CLS = 0
        POLY_CONS_HYB2 = 0
        POLY_CONS_EXT2 = 0
        POLY_CONS_TYP2 = 0 
        NATTR2 = 0
        EATTR2 = 0
        NNEATTR2 = 0
        SIM_ATTR2 = 0
        STRU_ATTR2 = 0
        COER_ATTR2 = 0 
        OTHER_ATTR2 = 0
        DIFF2 = [0, 0, 0]
        POLY_HYBS = []
        STA_CLSS = []
        YES_SITE0 = 0
        NOMINAL_ATTR0 = 0
        UNION_SITE0 = 0
        GUARDED_SITE0 = 0
        REFINE_SITE0 = 0
        CONMIS = 0
        MIS = 0
        with open(f'result_text/q5/table_{now}.txt', 'w+') as fw:
            for repo in repos:
                with open(f'result_text/q5/{repo}_mis.txt', 'r') as f:
                    lines = f.readlines()
                    CONMIS += int(lines[0].strip())
                    MIS += int(lines[1].strip())

                with open(f'result_text/q5/{repo}_cons.txt', 'r') as f:
                    lines = f.readlines()
                    YES_SITE0 += int(lines[0].strip())
                    NOMINAL_ATTR0 += int(lines[1].strip())
                    UNION_SITE0 += int(lines[2].strip())
                    GUARDED_SITE0 += int(lines[3].strip())
                    REFINE_SITE0 += int(lines[3].strip())
                    
                with open(f'result_text/q5/{repo}.txt', 'r') as f:
                    lines = f.readlines()
                    POLY_ATTR += int(lines[0].strip())
                    ALL_ATTR += int(lines[1].strip())
                    YES_SITE += int(lines[2].strip())
                    NOMINAL_ATTR += int(lines[3].strip())
                    UNION_SITE += int(lines[4].strip())
                    GUARDED_SITE += int(lines[5].strip())
                    GUARDED_SITE_AGGRE += int(lines[6].strip())
                    REFINE_SITE += int(lines[7].strip())
                    add_list(SENSI_SITE1, json.loads(lines[8].strip()))
                    SENSI_SITE2 += int(lines[9].strip())
                    SENSI_SITE3 += int(lines[10].strip())
                    SENSI_SITE4 += int(lines[11].strip())
                    CAUSE_ATTR0 += int(lines[12].strip())
                    CAUSE_ATTR1 += int(lines[13].strip())
                    CAUSE_ATTR2 += int(lines[14].strip())
                    # assert len(lines) == 27
                    if int(lines[2].strip()) != 0:
                        
                    # fw.write(f'{repo} & {sci_not(lines[9].strip())} & {sci_notm(lines[10].strip())}  & {sci_notm(lines[11].strip())}  & {sci_notm(lines[12].strip())} & {" & ".join([str(sci_notm(x)) for x in json.loads(lines[13].strip())])} & {" & ".join([str(sci_notm(x)) for x in json.loads(lines[14].strip())])} & {sci_notm(lines[15].strip())} \\\  \n')
                        fw.write(f'{repo} & {lines[1].strip()} & {lines[0].strip()}  & {lines[2].strip()}  & {sci_notm(int(lines[4].strip())/int(lines[2].strip()))} & {sci_notm(int(lines[5].strip())/int(lines[2].strip()))}  & {sci_notm(int(lines[7].strip())/int(lines[2].strip()))} & {sci_notm(int(lines[13].strip())/int(lines[2].strip()))} & {sci_notm(int(lines[12].strip())/int(lines[2].strip()))} & {sci_notm(int(lines[14].strip())/int(lines[2].strip()))}  \\\  \n')
                    
                with open(f'result_text/q5/{repo}_stat.txt', 'r') as f:
                    lines = f.readlines()
                    POLY_CONS2 += int(lines[0].strip())
                    POLY_HYB += int(lines[1].strip())
                    POLY_HYBS.append(int(lines[1].strip())/CLS_NUM[repo])
                    STA_CLS += int(lines[2].strip())
                    STA_CLSS.append(int(lines[2].strip())/CLS_NUM[repo])
                    POLY_CONS_HYB2 += int(lines[3].strip())
                    POLY_CONS_EXT2 += int(lines[4].strip())
                    POLY_CONS_TYP2 += int(lines[5].strip())
                    NATTR2 += int(lines[6].strip())
                    EATTR2 += int(lines[7].strip())
                    NNEATTR2 += int(lines[8].strip())
                    SIM_ATTR2 += int(lines[9].strip())
                    STRU_ATTR2 += int(lines[10].strip())
                    COER_ATTR2 += int(lines[11].strip())
                    OTHER_ATTR2 += int(lines[12].strip())
                    add_list(DIFF2, json.loads(lines[13].strip()))
            fw.write(f'{NAME[now+1]} & {ALL_ATTR} & {POLY_ATTR} & {YES_SITE} & {sci_notm(UNION_SITE/YES_SITE)} & {sci_notm(GUARDED_SITE/YES_SITE)} & {sci_notm(REFINE_SITE/YES_SITE)}  & {sci_notm(CAUSE_ATTR1/YES_SITE)}  & {sci_notm(CAUSE_ATTR0/YES_SITE)}  & {sci_notm(CAUSE_ATTR2/YES_SITE)} \\\\ \n')
        
            # fw.write(f'average & {sci_not(POLY_ATTR/ALL_ATTR)} & {sci_notm(NOMINAL_ATTR/YES_SITE)} & {sci_notm(UNION_SITE/YES_SITE)} & {sci_notm(GUARDED_SITE/YES_SITE)} & {sci_notm(GUARDED_SITE_AGGRE/YES_SITE)} & {sci_notm(REFINE_SITE/YES_SITE)}  & {" & ".join([str(sci_notm(x/YES_SITE)) for x in SENSI_SITE1])} & {sci_notm(SENSI_SITE2/YES_SITE)} & {sci_notm(SENSI_SITE3/YES_SITE)} & {sci_notm(SENSI_SITE4/YES_SITE)} \\\\ \\hline \n')
            # fw.write(f'average & {sci_not(ALL_ATTR)} & {sci_notm(NOMINAL_ATTR/YES_SITE)} & {sci_notm(UNION_SITE/YES_SITE)} & {sci_notm(GUARDED_SITE/YES_SITE)} & {sci_notm(GUARDED_SITE_AGGRE/YES_SITE)} & {sci_notm(REFINE_SITE/YES_SITE)}  \\\\ \\hline \n')
        # with open(f'result_text/q4f/overall.txt', 'w+') as f:
        #     f.write('TYPE' + '\t' + str(POLY_CONS_TYP2) + '\n')
        #     f.write('ATTR' + '\t' + str(POLY_CONS_EXT2) + '\n')
        #     f.write('HYB' + '\t' + str(POLY_CONS_HYB2) + '\n')
        with open(f'result_text/q2f/type_relation.txt', 'w+') as f:
            f.write(str(EATTR2) + '\n')
            f.write(str(NATTR2) + '\n')
            f.write(str(SIM_ATTR2) + '\n')
            f.write(str(COER_ATTR2) + '\n')
            f.write(str(STRU_ATTR2) + '\n')
            f.write(str(OTHER_ATTR2) + '\n')

        # SENSI_SITE2 = SENSI_SITE2[-1:]
        f4.write(f'{NAME[now+1]}  & {sci_not(STA_CLS/ALL_OOM["override"][3])} & {sci_not(POLY_HYB/ALL_OOM["override"][3])} & {sci_not(average(STA_CLSS))} & {sci_not(average(POLY_HYBS))} \\\\ \n')
        f5.write(f'{NAME[now+1]} & {ALL_ATTR} & {POLY_ATTR} & {YES_SITE} & {sci_notm(UNION_SITE/YES_SITE)} & {sci_notm(GUARDED_SITE/YES_SITE)}  & {sci_notm(REFINE_SITE/YES_SITE)} & {sci_notm(CAUSE_ATTR1/YES_SITE)}  & {sci_notm(CAUSE_ATTR0/YES_SITE)}  & {sci_notm(CAUSE_ATTR2/YES_SITE)} & {MIS} & {CONMIS}\\\\ \n')
        
        # print(YES_SITE)
