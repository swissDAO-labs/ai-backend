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

