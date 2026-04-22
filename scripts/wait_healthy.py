#!/usr/bin/env python3
"""Thin wrapper so scripts/wait_healthy.py matches scripts/serve_vllm.sh."""
from agentdiet.cli.health import main

raise SystemExit(main())
