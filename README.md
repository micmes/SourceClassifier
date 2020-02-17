# SourceClassifier
An algorithm to classify the sources in the 4FGL catalog

[![Build Status](https://travis-ci.com/micmes/SourceClassifier.svg?branch=master)](https://travis-ci.com/micmes/SourceClassifier)

[![Documentation Status](https://readthedocs.org/projects/sourceclassifier/badge/?version=latest)](https://sourceclassifier.readthedocs.io/en/latest/?badge=latest)

## First install

### for Windows user
We did not built a windows version of this package yet. So, we recommend to follow the instruction at https://docs.microsoft.com/it-it/windows/wsl/install-win10 to install a Linux environment (Ubuntu is preferred). After that, open bash and proceed with the following operations. 

### for Ubuntu user
Create a virtual environment and install all the required packages running the following lines:
```
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```
It is strictly recommended to check for new packages once in a while, running the third line. 

## Usage
Every time you want to use this tool, you have to run
```
source venv/bin/activate
```
in your bash first. To see some example of package at work we refer to the documentation:
https://sourceclassifier.readthedocs.io/en/latest/?badge=latest

## Uninstall 
Simply delete 'SourceClassifier' folder. 

## Developer options

### Testing
In order to setup the environment variables, run 
```
source setup.sh
```
and then, to run tests, just 
```
cd tests
make
```
