.PHONY: help serve stop status health pilot gate pilot-full pilot-clean \
        collect collect-report collect-clean test smoke

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

help:
	@echo "vLLM serving:"
	@echo "  serve            Start vLLM server in the background"
	@echo "  stop             Stop vLLM server (via pid file)"
	@echo "  status           Report vLLM server status"
	@echo "  health           Wait until vLLM /models endpoint is 200 (timeout 180s)"
	@echo ""
	@echo "Gate 1 (pilot):"
	@echo "  pilot            Run 30-question single vs 3x3 debate pilot (resumable)"
	@echo "  gate             Build Gate-1 report; exit 0 pass / 10 soft-fail / 20 hard-fail"
	@echo "  pilot-full       serve -> health -> pilot -> gate, one shot"
	@echo "  pilot-clean      Remove artifacts/pilot/ (not the LLM cache)"
	@echo ""
	@echo "Full collection (100Q):"
	@echo "  collect          Run 100-question 3x3 debate collection (resumable)"
	@echo "  collect-report   Print existing collection manifest summary"
	@echo "  collect-clean    Remove artifacts/dialogues/ (not the LLM cache)"
	@echo ""
	@echo "Tests:"
	@echo "  test             Full pytest suite (no network)"
	@echo "  smoke            Smoke-tagged tests only"

serve:
	scripts/serve_vllm.sh start

stop:
	scripts/serve_vllm.sh stop

status:
	scripts/serve_vllm.sh status

health:
	$(PYTHON) scripts/wait_healthy.py --timeout 180

pilot:
	$(PYTHON) -m agentdiet.cli.pilot

gate:
	$(PYTHON) -m agentdiet.cli.gate --report

pilot-full: serve health pilot gate

pilot-clean:
	rm -rf artifacts/pilot artifacts/pilot_report.md

collect:
	$(PYTHON) -m agentdiet.cli.collect

collect-report:
	$(PYTHON) -m agentdiet.cli.collect --report-manifest

collect-clean:
	rm -rf artifacts/dialogues

test:
	$(PYTEST) tests/ --timeout=30

smoke:
	$(PYTEST) -m smoke --timeout=30
