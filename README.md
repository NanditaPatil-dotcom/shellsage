
# ShellSage 

ShellSage is your friendly command-line companion — run any shell command with beautiful output formatting.
But a condition, it runs only on python envs

### Installation(only on python env):
```bash
pip install shellsage-cli
```

### Usage:
```bash
shellsage "ls -la"
shellsage "ps aux | grep python"
shellsage "find . -name '*.py' -exec wc -l {} +"

