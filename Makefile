.PHONY: new_env
new_env:
	@[ -n "$$(poetry run which python)" ] && source $$(poetry env info --path)/bin/activate && poetry env remove $$(which python) || true
	poetry install
	@[ -n "${VIRTUAL_ENV}" ] || exec poetry shell


CLEAN_PATTERNS := $(shell awk '/^# Clean$$/{f=1; next} /^# End Clean$$/{f=0} f' .gitignore)

.PHONY: clean
clean:
	@$(foreach pattern,$(CLEAN_PATTERNS), \
		find . -name "$(pattern)" -exec rm -rf {} +; \
	)

.PHONE: requirements
requirements:
	@poetry export -f requirements.txt --output simple-frontend/requirements.txt --without-hashes --only frontend
	@poetry export -f requirements.txt --output microservices/stable-diffusion/requirements.txt --without-hashes --only stable-diff

.PHONY: format
format:
	@poetry run ruff check --select I --fix .
	@poetry run ruff format .

.PHONY: lint
lint:
	@poetry run ruff check .
