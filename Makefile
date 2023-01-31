.PHONY: env
env:
	mamba create -n absolv
	mamba env update -n absolv --file devtools/envs/base.yaml
	conda run --no-capture-output --name absolv pip install --no-deps -e .
	conda run --no-capture-output --name absolv pre-commit install

.PHONY: format
format:
	pre-commit run --all-files