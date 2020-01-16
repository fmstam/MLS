__package__ = 'MLS'
import sys
sys.path.append("..")

from MLS.examples.EnvEmptySlot import EnvEmptySlot

# state and action
state_size = 5
action_space = [i for i in range(state_size)]

def main():
    env = EnvEmptySlot(state_size=state_size, action_space=action_space)
    env.step(2)
    print(env.play_ground)


if __name__ == "__main__":
    # execute only if run as a script
   main()
