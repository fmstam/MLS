# a learning class of simpy
import simpy
import random

FRAME_LENGTH = 10000 # 10 ms
CCA_TIME = 3        # 20 us
IDLE_TIME = 50       # 5% FRAME_LENGTH
BACK_OFF_TIME = FRAME_LENGTH
MAX_ATTEMPTS = 10 # before declaring failure
is_channel_free = True # free



def node(env, name, bcs):
    global is_channel_free
    # arrival time
    arrival_time = random.expovariate(1. / 2.) # possiion arrival rate every 10 s
    yield env.timeout(arrival_time)
    print('node %s has arrived at %d' % (name, env.now))

    # maximum attempts to access the channel after failure
    attempts = MAX_ATTEMPTS
    while attempts > 0:
        with bcs.request() as req:
            print('node %s is trying to access the channel at %s' %(name, env.now))

            # CCA period
            cca = 0
            cca_time = env.now
            while cca < CCA_TIME and is_channel_free == True:
                print('node %s is_channel_free %d at %s' % (name, is_channel_free, env.now))
                yield env.timeout(1)
                cca += 1
            print('node %s has made CCA for %dus at time %s ' % (name, cca, cca_time ))
            
            res = yield req | env.timeout(0)
            # send or backoff period
            if req in res:
                is_channel_free = False # channel is occuped
                print('channel is free for node %s at time %s' % (name, env.now))
                # Frame sending
                print('node %s starting  sending at %s' % (name, env.now))
                yield env.timeout(FRAME_LENGTH)
                # done
                print('node %s finished sending at %s' % (name, env.now))
                attempts = -1
                is_channel_free = True
            else:
                attempts -= 1
                print('node %s is backing off at %s for %d us ' % (name, env.now, BACK_OFF_TIME))
                yield env.timeout(BACK_OFF_TIME)

    if attempts >= 0:
        print('node %s coulde not access the channel' % name)


env = simpy.Environment()
bcs = simpy.Resource(env, capacity=1)
for i in range(2):
     env.process(node(env, '%d' % i, bcs))
    
env.run()
