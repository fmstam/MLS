# %% main file, i.e., runner
import sys
sys.path.append("..")

from MLS.examples import scenario_ddpg as s
from MLS.torchDRL.TrainingManager import TrainingManager as TM

import timeit



def main():
    # define a training manager object
    tm = TM(s.num_episodes, 
            s.episode_length, 
            s.agent,
            s.env,
            log_file=s.log_file)

    print('Scenario:%s' % s.title)
    start = timeit.default_timer()
    # let it do the magic
    tm.run(verbose=True)
    end = timeit.default_timer()
    print('\n It took ~{} useconds'.format(str(round(end-start))))
    


    

if __name__ == "__main__":
    # execute only if run as a script
   main()


# %%
