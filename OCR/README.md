Create and prepare venv:
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
Install and register kernel:
./.venv/bin/pip install ipykernel
./.venv/bin/python -m ipykernel install --user --name bajaj-finserv --display-name "Python (.venv) bajaj-finserv"
Install project requirements:
./.venv/bin/pip install -r requirements.txt