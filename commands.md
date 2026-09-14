# Useful commands

## monitor model performance every X sec

```bash
watch -n 1 -d 'ollama ps; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader'

# or without refreshing the terminal
while true; do
    date '+%Y-%m-%d %H:%M:%S'
    ollama ps
    nvidia-smi \
        --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    printf '\n'
    sleep 0.1
done
```