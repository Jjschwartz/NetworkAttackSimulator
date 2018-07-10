# Cyber Attack Simulator

A environment for testing AI agents against a simple simulated computer network with subnetworks and machines with different vulnerabilities.

The aim is to retrieve sensitive documents located on certain machines on the network without being caught. Each document provides a large reward to the agent.

The attack terminates in one of the following ways:
1. the agent collects all sensitive documents = goal
2. the agent get caught = loss

The agent receives information about the network topology, specifically:
1. the machines and subnets in the network
2. which machines have the sensitive documents

The agent does not know which services are running on which machine, and hence which machines are vulnerable to which exploits.

The actions available to the agent are exploits and scan.
- exploits:
    - there is one exploit action for each possible service running, which is a envirnment parameter
- scan:
    - reveals which services are present and absent on a target machine (i.e. inspired by Nmap behaviour)

Each action must be launched against a specific machine, but actions will only possibly work on machines that are reachable.

A machine is reachable if:
1. it is on exposed subnet (subnet 1 by default) (i.e. this would be the machines available to public, e.g. webserver)
2. it is on a subnet reachable from a machine on connected subnet that has been successfuly compromised by agent

# Dependencies
- Python >3.5
- Numpy

## Other future dev todo
1. Action space class
    1. to allow for random_choice function
    2. optimized data structure?

## Ideas for future environment features
1. Attack abort option
    * Agent can choose to abort attack in order to avoid detection
    * Ends episode with agent recieveing only gained rewards
    * Would be a sub-optimal goal
1. Non-deterministic actions
    * exploits may fail with some probability, even when service is present on target machine
1. Each action produces a certain amount of noise which eventually leads to being caught
2. Add support for more than 3 subnets and users to specify depth (i.e. number of subnets to traverse to reach goals)
