# Cyber Attack Simulator

A environment for testing AI agents against a simulated Network.

## Ideas for future development
1. Attack abort option
    * Agent can choose to abort attack in order to avoid detection
    * Ends episode with agent recieveing only gained rewards
    * Would be a sub-optimal goal
1. Non-deterministic actions
    * exploits may fail with some probability, even when service is present on target machine
1. Each action produces a certain amount of noise which eventually leads to being caught
2. Add support for more than 3 subnets and users to specify depth (i.e. number of subnets to traverse to reach goals)
