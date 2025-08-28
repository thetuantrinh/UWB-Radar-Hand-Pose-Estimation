#! /bin/bash
echo "start"
sleep 2
while true; do
    wandb sync ./wandb/offline-run-20230711*
    echo "syncing wandb logs to the cloud after 300 seconds"
    sleep 300
done

