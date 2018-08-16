#!/bin/bash

# Intended to reset ECS agent state
sleep 5
if ( sudo status ecs | grep start ); then
    sudo stop ecs
    sudo rm -rf /var/lib/ecs/data/ecs_agent_data.json
fi
