from cyber_attack_simulator.envs.pomdp_generator import generate_pomdp_config


config_path = "/home/jonathon/Documents/Uni/COMP6801/CyberAttackSimulator/cyber_attack_simulator/configs/"
small_config = config_path + "small.yaml"
small2_config = config_path + "small2.yaml"
small_med_config = config_path + "small_med.yaml"
med_config = config_path + "medium.yaml"


def main():
    generate_pomdp_config(small_config)


if __name__ == "__main__":
    main()
