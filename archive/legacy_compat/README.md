Archived legacy compatibility notes.

These files document compatibility layers removed from the active pipeline.
They are kept only for reference when reading old multi-agent outputs or
migrating archived JSON. Current code uses only:

- bull_agent
- bear_agent
- target_args

Additional archived code:

- `anthropic_compatible.py`: old DeepSeek/Anthropic-compatible provider.
- `biode/`: standalone older Bi-ODE experiment package, not imported by the current pipeline.
