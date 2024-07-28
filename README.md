# Exploring Jax

# Getting Started

## Prerequisites

* Python 3.12
* UV

### Apple Silicon

I had to pin libomp to 11.1.0 to avoid segfaults in pytorch.

```bash
curl -sL -o libomp.rb https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew install ./libomp.rb
```

## Configure Environment

```bash
# Configure environment
source environment.sh
make

# Activate venv
source .venv/bin/activate

# Launch jupyter
jupyter lab
```
