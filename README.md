# SourceClassifier
An algorithm to classify the sources in the 4FGL catalog

[![Build Status](https://travis-ci.com/micmes/SourceClassifier.svg?branch=master)](https://travis-ci.com/micmes/SourceClassifier)

## Initial setup (Windows)
Follow https://docs.microsoft.com/it-it/windows/wsl/install-win10 and then, after opening bash, proceed with _Initial setup (Ubuntu)_.

*

## Initial setup (Ubuntu)
The first time, in order to create a virtual environment and install all the required packages, run the following lines:
```
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```
it is strictly recommended to check for new packages once in a while, running the third line. From now on, on reboot, just type:
```
source venv/bin/activate
```
to activate virtualenv.

### Environment variables
In order to setup the environment variables, run 
```
source setup.sh
```
every time on reboot. 

## Uninstall 
Simply delete 'SourceClassifier' folder. 

## Developer options

### Testing
To run tests, just 
```
cd tests
make
```
