# ANN Experiments

This is a repo for quickly prototyping features for the ANN index.

## Retrieving Data Sets

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
>>> dvc pull data queries-quora-E5-small.fvec.dvc
>>> dvc pull data corpus-quora-E5-small.fvec.dvc
```

## Mac Setup

### Xcode

Building requires Apple's development environment, Xcode, which can be downloaded from
<https://developer.apple.com/download/>. You will need to register as a developer with Apple. Alternatively,
you can get the latest version of Xcode from the App Store.

For C++17 at least Xcode 10 is required, and this requires macOS High Sierra or above. using Monterey or
Ventura, you must install Xcode 14.2.x or above. Xcode is distributed as a `.xip` file; simply double click
the `.xip` file to expand it, then drag `Xcode.app` to your `/Applications` directory.

There are no command line tools out-of-the-box, so you'll need to install them following installation of
Xcode. You can do this by running:

```
xcode-select --install
```

at the command prompt.

### CMake

Download the graphical installer for version 3.23.2 from <https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-macos-universal.dmg> (or get a more recent version).

Open the `.dmg` and install the application it by dragging it to the `Applications` folder.

Then make the `cmake` program accessible to programs that look in `/usr/local/bin`:

```
sudo mkdir -p /usr/local/bin
sudo ln -s /Applications/CMake.app/Contents/bin/cmake /usr/local/bin/cmake
```

## Building

To build native code navigate to the root directory and run:

```
>>> cmake -S . -B build
>>> cmake --build build
```

This should create an executable in `build/run_quantisation`.

You can run help on this and you to see the options:
```
>>> ./build/run_quantisation -h
run_quantisation [-h,--help] [-u,--unit] [-s,--scalar] [--scann] [-r,--run DATASET] [-m, --metric METRIC]
	--help		Show this help
	--unit		Run the unit tests (default false)
	--scalar N	Use 4 or 8 bit scalar quantisation (default None)
	--scann		Use anisotrpoic loss when building code books (default false)
	--run DATASET	Run a test dataset
	--metric METRIC	The metric, must be cosine or dot, with which to compare vectors (default cosine)
```
