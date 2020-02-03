# SourceClassifier
An algorithm to classify the sources in the 4FGL catalog

## Initial setup (Windows)
Follow https://docs.microsoft.com/it-it/windows/wsl/install-win10 and then, after opening bash, proceed with _Initial setup (Ubuntu)_.

*

## Initial setup (Ubuntu)
To create a virtual environment and install all the required packages (the first line can be runned the first time only):
```
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

In order to setup the environment variables, one can run alternatively 
```
bash setup_u.sh
```
which adds the project root to the python environment (once for all), or 
```
source setup.sh
```
which requires to run code from inside that shell (and this operation should be done every time on reboot). 


