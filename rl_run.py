import sys
sys.path.append("..")

from MLS.examples.all_one import scenario as s
from MLS.torchDRL.TrainingManager import TrainingManager as TM



def main():

    # define a training manager object
    tm = TM(s.num_episodes, 
            s.episode_length, 
            s.agent,
            s.env,
            device=s.device,
            log_file=s.log_file)

    # let it do the magic
    tm.run()

    

if __name__ == "__main__":
    # execute only if run as a script
   main()
