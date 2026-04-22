.PHONY: help serve stop status health pilot gate pilot-full pilot-clean \
        collect collect-report collect-clean \
        extract extract-report extract-clean spot-check \
        analyze analyze-report analyze-clean \
        ablate ablate-report ablate-clean gate2 \
        policy-sample \
        evaluate evaluate-report evaluate-clean \
        test smoke

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
	@echo "Claim extraction:"
	@echo "  extract          Extract 6-type claims from all collected dialogues (resumable)"
	@echo "  extract-report   Print existing claim-extraction manifest summary"
	@echo "  extract-clean    Remove artifacts/claims/ (not cache or dialogues)"
	@echo "  spot-check       Sample K dialogues (default 10) -> spot_check.csv"
	@echo ""
	@echo "Analysis (flip events + per-claim signals):"
	@echo "  analyze          Compute flip_events.jsonl + signal_scores.parquet"
	@echo "  analyze-report   Print existing analysis manifest summary"
	@echo "  analyze-clean    Remove artifacts/analysis/ (not cache / dialogues / claims)"
	@echo ""
	@echo "Type-level ablation (Gate 2):"
	@echo "  ablate           Per-type ablation on single_wrong AND debate_right subset"
	@echo "  ablate-report    Print existing ablation summary"
	@echo "  ablate-clean     Remove ablation artifacts only"
	@echo "  gate2            Emit gate2_report.md; exit 0 PASS / 10 null / 20 inconclusive"
	@echo ""
	@echo "Compression policy (Day 3):"
	@echo "  policy-sample    Seed artifacts/compression/policy.json from the template (no-clobber)"
	@echo ""
	@echo "Evaluation sweep:"
	@echo "  evaluate         5 methods x N questions -> results.json + invariant violations"
	@echo "  evaluate-report  Print per-method summary"
	@echo "  evaluate-clean   Remove artifacts/evaluation/ only"
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

extract:
	$(PYTHON) -m agentdiet.cli.extract

extract-report:
	$(PYTHON) -m agentdiet.cli.extract --report-manifest

extract-clean:
	rm -rf artifacts/claims

spot-check:
	$(PYTHON) -m agentdiet.cli.spot_check --k 10

analyze:
	$(PYTHON) -m agentdiet.cli.analyze

analyze-report:
	$(PYTHON) -m agentdiet.cli.analyze --report

analyze-clean:
	rm -rf artifacts/analysis

ablate:
	$(PYTHON) -m agentdiet.cli.ablate

ablate-report:
	$(PYTHON) -m agentdiet.cli.ablate --report

ablate-clean:
	rm -f artifacts/analysis/ablation.jsonl artifacts/analysis/ablation_summary.json artifacts/analysis/ablation_manifest.json artifacts/analysis/gate2_report.md

gate2:
	$(PYTHON) -m agentdiet.cli.ablate --gate2

policy-sample:
	mkdir -p artifacts/compression && cp -n docs/policy.sample.json artifacts/compression/policy.json || true

evaluate:
	$(PYTHON) -m agentdiet.cli.evaluate

evaluate-report:
	$(PYTHON) -m agentdiet.cli.evaluate --report

evaluate-clean:
	rm -rf artifacts/evaluation

test:
	$(PYTEST) tests/ --timeout=30

smoke:
	$(PYTEST) -m smoke --timeout=30
