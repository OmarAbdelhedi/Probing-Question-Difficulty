# Resolve paths relative to THIS script, not the caller's CWD, so sourcing from
# any directory (src/, scripts/, repo root) finds the same repo-root .env.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-/scratch/USER/NeoDip_env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$SCRIPT_DIR/../requirements.txt}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../.env}"

load_modules() {
    module load StdEnv/2023 gcc arrow/18.1.0 python/3.11.5 opencv rust
    module load r-bundle-bioconductor/3.21
    module load Jinja2/3.1.2 igraph/2.2.0 pillow/12.1.0
    module load cuda/12.2 cudnn/8.9.5.29 tensorrt/8.6.1.6
    module load exceptiongroup/1.2.2 cycler/0.12.1 cffi/2.0.0
    module load pyzmq/27.1.0 psutil/7.2.2 pip/26.0.1 idna/3.11
    module load python-dateutil/2.8.2 importlib_metadata/8.7.1
    module load certifi/2026.2.25 httpx/0.28.1 matplotlib/3.10.8
    module load platformdirs/4.3.8 py-cpuinfo/9.0.0 PyYAML/6.0.1
    module load contourpy/1.3.1 cryptography/46.0.3 distlib/0.4.0
    module load msgpack/1.1.2 pbr/6.1.0 tzdata/2025.3 distro/1.9.0
    module load requests/2.32.5 rich/14.3.2 tomli/2.2.1 zipp/3.23.0
    module load six/1.17.0 scipy/1.17.0 pyparsing/3.3.2 pytz/2025.2
    module load pandas/3.0.0 wheel/0.41.2 urllib3/2.6.3 sympy/1.13.3
    module load typing_extensions/4.15.0 sniffio/1.3.0 kiwisolver/1.4.9
    module load igraph/0.10.13
    module load python-igraph/0.11.8
    module load packaging/25.0
}

set_env_vars() {
    export RUSTC_BOOTSTRAP=1
    export PYTHONPATH=$(pwd):$PYTHONPATH
    export VLLM_DISTRIBUTED_TIMEOUT_MINUTES=20
    export HF_HUB_DISABLE_XET=1
    export OMP_NUM_THREADS=1
    export VLLM_USE_FLASHINFER_SAMPLER=1

    local _outlines_suffix="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_$$"
    export OUTLINES_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/outlines_cache_${_outlines_suffix}"
    mkdir -p "$OUTLINES_CACHE_DIR"
    echo "  -> OUTLINES_CACHE_DIR:       $OUTLINES_CACHE_DIR"
}

# ==========================================
# Runtime config (applied every run)
# ==========================================
set_runtime_config() {
    echo "==> Setting runtime config (vLLM, wandb, secrets)"

    export VLLM_LOGGING_LEVEL=DEBUG
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"

    export WANDB_DISABLE_CODE=true
    export WANDB__DISABLE_META=true

    # Load secrets from a gitignored .env file:
    #   HF_TOKEN=...
    #   WANDB_API_KEY=...
    #   OPENROUTER_API_KEY=...
    if [ -f "$ENV_FILE" ]; then
        echo "==> Loading secrets from $ENV_FILE"
        set -a
        # Strip CRs so a Windows-edited (CRLF) .env doesn't append \r to secrets.
        # shellcheck disable=SC1090
        source <(sed 's/\r$//' "$ENV_FILE")
        set +a
    else
        echo "    (no $ENV_FILE found — relying on inherited environment for secrets)"
    fi

    # Make OPENAI_API_KEY available to OpenAI-compatible clients in api mode.
    # If only OPENROUTER_API_KEY is set, mirror it into OPENAI_API_KEY.
    if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ]; then
        export OPENAI_API_KEY="$OPENROUTER_API_KEY"
    fi

    if [ -n "${WANDB_API_KEY:-}" ] && [ "$WANDB_API_KEY" != "XXXX" ]; then
        wandb login --relogin "$WANDB_API_KEY"
    else
        echo "    (skipping wandb login: WANDB_API_KEY not set)"
    fi
}

# ==========================================
# Build helpers
# ==========================================
ensure_repo() {
    # ensure_repo <dir> <git-url> [<checkout-ref>]
    local dir="$1" url="$2" ref="${3:-}"
    if [ ! -d "$dir/.git" ]; then
        echo "==> Cloning $url into $dir"
        git clone "$url" "$dir"
    else
        echo "==> Reusing existing $dir"
    fi
    if [ -n "$ref" ]; then
        ( cd "$dir" && git fetch --tags --quiet && git checkout "$ref" )
    fi
}

build_inplace_abn() {
    ensure_repo inplace_abn https://github.com/mapillary/inplace_abn.git
    ( cd inplace_abn && pip install . --no-build-isolation )
}

build_kornia_rs() {
    ensure_repo kornia-rs https://github.com/kornia/kornia-rs.git v0.1.7
    (
        cd kornia-rs
        git submodule update --init --recursive
        cd kornia-py
        PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install .
    )
}

create_env() {
    echo "==> (Re)creating environment at: $ENV_DIR"

    if [ -d "$ENV_DIR" ]; then
        echo "==> Removing existing environment at $ENV_DIR"
        rm -rf "$ENV_DIR"
    fi

    rm -rf ~/.cache/pip ~/.cache/huggingface ~/.local/share/virtualenv ~/.cache/flashinfer

    load_modules

    mkdir -p "$(dirname "$ENV_DIR")"
    virtualenv --python=python3.11 "$ENV_DIR"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"

    pip cache purge
    pip install --upgrade pip

    set_env_vars

    # Run all clones/builds from a known working dir.
    local build_root="${SLURM_TMPDIR:-/tmp}/build"
    mkdir -p "$build_root"
    pushd "$build_root" > /dev/null

    # PyTorch (CUDA 12.9 wheels — works on cuda/12.2 module)
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

    build_inplace_abn
    build_kornia_rs

    # Core Python deps
    pip install datasets==2.19.0
    pip install mistral-common==1.11.0 --no-deps
    pip install wandb==0.24.0+computecanada
    pip install vllm
    pip install xformers==0.0.28.post3 --no-build-isolation
    pip install triton==3.2.0 --no-deps
    pip install transformers

    # Repo requirements (no deps to avoid clobbering CC-pinned versions)
    pip install --no-deps --no-cache-dir -r "$REQUIREMENTS_FILE"

    # NeoDip-specific deps
    pip install weave
    pip install diplomacy
    pip install jsonschema

    # Optional / platform-extras (kept from your raw script)
    pip install libtpu || echo "    (libtpu install failed — skipping; not needed on NVIDIA H100)"
    pip install amdsmi || echo "    (amdsmi install failed — skipping; not needed on NVIDIA H100)"

    popd > /dev/null

    # Remove broken CC pip dist-info that crashes importlib_metadata
    echo "==> Cleaning broken CC pip dist-info"
    find "$ENV_DIR/lib/python3.11/site-packages/" \
        -maxdepth 1 -name 'pip-*+computecanada.dist-info' \
        -exec rm -rf {} +

    touch "$ENV_DIR/.build_complete"

    echo "==> Environment created successfully at $ENV_DIR"
}

activate_env() {
    if [ ! -f "$ENV_DIR/.build_complete" ]; then
        echo "ERROR: Environment build did not complete at $ENV_DIR"
        echo "       Run this script with --create to build it:"
        echo "         source set_up_env_cc.sh --create"
        return 1
    fi

    echo "==> Loading modules"
    load_modules

    echo "==> Activating environment at $ENV_DIR"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"

    set_env_vars
    set_runtime_config

    export HF_HOME=/scratch/USER/huggingface-cache

    echo "==> Environment ready"
}

# ==========================================
# Entry point
# ==========================================
# Usage:
#   source set_up_env_cc.sh              # just activate the existing env
#   source set_up_env_cc.sh --create     # (re)create the env, then activate

DO_CREATE=0
for arg in "$@"; do
    case "$arg" in
        --create) DO_CREATE=1 ;;
        *) echo "Unknown argument: $arg" ;;
    esac
done

if [ "$DO_CREATE" -eq 1 ]; then
    create_env
fi

activate_env