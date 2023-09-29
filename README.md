 This is the artifact of a study on Python objects. 

## Structure
After unzip, the artifact contains two folders: 
purepython: The instrumented CPython 3.9 interpreter
python-study: The dynamic analysis and other experimental scripts 


## Installation

### Build the Interpreter
Before building the interpreter, you must change to the purepython folder. 

Since the modified interpreter is mostly similar to the original interpreter, you can build it following the offical instruction (<https://github.com/python/cpython>). 
After the interpreter is built, create a virtual environment for it. The projects used in the experiment would be installed to the virtual environment. 

### Install the Dynamic Analysis and Other Experimental Scripts 

Those scripts are written in Python. 
You can create a new conda environment to run them. 
```
conda create -n script Python=3.9
```

Next, install those needed libraries: 
* matplotlib
* numpy
* lark

## Usage

We do not provide all the collected traced files in this artifact (the size of all trace files is beyond 100G!), but we do provide the files used in the last step to get the statistics reported in the paper. 

If you just want to evaluate the results, you can directly go to the last step (Get the Statistics). 
If you want to collect the traces, run the dynamic analysis by yourself, you can follow the listed steps sequentially. 

### Collect Traces
To collect the traces for the testing process of one project, say rich, you must first get the source code of the project . 
```
git clone https://github.com/Textualize/rich
```

Then, install the project in the virtual environment previously built. Suppose you have opened this environment in the current shell: 
```
cd rich
pip install .
```

Now, run the tests using the following command, and the collected traces would be stored to the pydyna folder in the root directory of the interpreter. 
```
python /Path_To_The_Dynamic_Analysis_Scripts/pre_run_biend.py rich 0 0 0
```

### Config the Dynamic Analysis. 
Open pre_run_biend.py in the python-study root. 

Config the input folder (the folder storing the traces) by:
```
changing all the /home/user/purepython/ to your interpreter path. 
```
Config the output folder (the folder to store the analyzed results) by changing this line: 
```
save_path = f'save_fast/{repo}_save_{c}' if repo in fast_repos else f'/data1/sk/repos/{repo}_save_{c}'
```
### Run the Dynamic Analysis in Parallel

Change into the python-study root. 
Run the dynamic analysis using the following command: 

This file contains three modes to run, i.e., 1 1 1, 1 0 1 and 1 0 0.
In the first run, use 1 1 1, which takes the original traces and does the analysis. This time, the analysis would process the raw traces to a two small intermediate representations and store them.  
In the following runs, you can use 1 0 1 to load the first intermediate representation and do the whole analysis, which will make the analysis faster. 
In the following runs, if you just want to do the class analysis, you can use 1 0 0.


```python
python worker.py
```

### Get the Statistics
After all the projects have been analyzed, get the result in the paper using the following command: 
```python
python summarizer.py
```
