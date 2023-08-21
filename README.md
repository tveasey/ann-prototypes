# ANN Experiments

This is a repo for quickly prototyping features for the ANN index.

First create a virtual Python 3 environment which we use for managing data and running tests:
```
>>> mkdir python && cd python && python -m venv `pwd` && source bin/activate
```

Next install dvc and pull test data sets:
```
>>> gcloud auth application-default login
>>> pip install dvc
>>> pip install dvc-gs
>>> dvc pull
```
Note that this requires gsutils and access to our gcp bucket. You can also download just a single data set using:
```
>>> dvc pull data dim-quora-E5-small.csv.bz2.dvc
>>> dvc pull data queries-quora-E5-small.csv.bz2.dvc
>>> dvc pull data corpus-quora-E5-small.csv.bz2.dvc
```

To build native code navigate to the root directory and run:
```
>>> cmake -S . -B build
>>> cmake --build build
```

This should create an executable in build/run_pq. You can run help on this and you should see:
```
>>> ./build/run_pq -h
run_pq [-h,--help] [-u,--unit] [-s,--smoke] [-r,--run DIR]
	--help		Show this help
	--unit		Run the unit tests
	--smoke		Run the smoke test
	--run DIR	Run the example from the gist in DIR
```
