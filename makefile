install: 
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest --nbval notebook/recommender_imbd.ipynb
	#python -m pytest -vvv --cov=greedy_coin coin/test_greedy_coin.py
	#python -m pytest -vvv --cov=get_long_lat_cities tests/test_get_long_lat_cities.py
	#python -m pytest -vvv --cov=tspEurope tests/test_tspEurope.py

format:
	black ./**/*.py


lint:
	pylint --disable=R,C ./**/*.py

all: install lint test format