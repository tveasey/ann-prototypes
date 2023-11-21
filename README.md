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

### Boost 1.83.0

Download version 1.83.0 of Boost from <https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.tar.bz2>. You must get this exact version, as the Machine Learning build system requires it.

Assuming you chose the `.bz2` version, extract it to a temporary directory:

```bash
bzip2 -cd boost_1_83_0.tar.bz2 | tar xvf -
```

In the resulting `boost_1_83_0` directory, run:

```bash
./bootstrap.sh --with-toolset=clang --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=mpi --without-libraries=python --without-icu
```

This should build the `b2` program, which in turn is used to build Boost.

To complete the build:

```bash
./b2 -j8 --layout=versioned --disable-icu cxxflags="-std=c++17 -stdlib=libc++" linkflags="-std=c++17 -stdlib=libc++ -Wl,-headerpad_max_install_names" optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC
sudo ./b2 install --layout=versioned --disable-icu cxxflags="-std=c++17 -stdlib=libc++" linkflags="-std=c++17 -stdlib=libc++ -Wl,-headerpad_max_install_names" optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC
```

to install the Boost headers and libraries.

## Building

To build native code navigate to the root directory and run:

```bash
>>> cmake -S . -B build
>>> cmake --build build
```

There are two targets `build/run_tests` and `build/run_benchmark`.

## Running Tests

Testing uses the boost test framework. After building you can run all the tests using

```bash
>>> ./build/run_tests
```

Individual tests can be run using for example

```bash
>>> ./build/run_tests --run_test=pq
```

Run `./build/run_tests --help` for more information.

## Running Benchmarks

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
