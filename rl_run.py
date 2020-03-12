# %% main file, i.e., runner
"""
This file is the entry point for running any scenario.
It requires two things: a scenario file and the training manager.
"""
import sys,os
sys.path.append('..')



from MLS.ceot_drl.scenarios import scenario_ddpg as s
from MLS.ceot_drl.core.TrainingManager import TrainingManager as TM

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
    tm.run(verbose=False)
    end = timeit.default_timer()
    print('\n It took ~{} useconds'.format(str(round(end-start))))
    


if __name__ == "__main__":
    # execute only if run as a script
   main()


# %%
