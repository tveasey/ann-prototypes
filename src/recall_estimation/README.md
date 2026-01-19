## Recall estimation via sampling

Here, we explore the accuracy with which we can estimate recall as a function of search
parameters by computing this quantity on different size samples of the data.

Specifically for ANN index settings $\theta$ we explore the relationship $f$ between
```math
\text{recall}(\theta| Q, D_s) = f(\text{recall}(\theta| Q, D))
```

Here, $D$ and $D_s$ denote the document set and sampled document set, respectively, and
$Q$ denotes a query set. In particular, we would like to determine if it is possible to
accurately estimate $f$ and so estimate recall on the full index using
$f^{-1}(\text{recall}(\theta| Q, D_s))$ if $|D_s| \ll |D|$.

To install dependencies in a virtual environment with uv:
```bash
brew install uv
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

To start elasticsearch run
```bash
./install-and-start-elastic-mac.sh
```
You also need to update the `.env` file to access from the experiment framework.