# a learning class of simpy
import simpy
import random

# sim params
FRAMES_TO_SIMULATE = 300
ACK_FAILURE_PROBABILITY = 0.1 # this paramater simulate the environement for ack not being recieved
                              # the higher this value the higher the packet drop rate

# MAC params
SYMBOL_TIME = 16  #us
SLOT_LENGTH = 20 * SYMBOL_TIME    # 20 symbols
FRAME_LENGTH = 10000  # 10 ms(as the paper)    10 * SLOT_LENGTH # 10 slots
CCA_TIME = 8 * SYMBOL_TIME       # 8 symbols
IDLE_TIME = 0.05 * FRAME_LENGTH       # 5% FRAME_LENGTH

# min and max BE in slots
MIN_BACKOFF_TIME = 1 
MAX_BACKOFF_TIME = 20
# maximum number of backoffs
MAX_BACKOFFS = 5 # before declaring failure

BACKOFF_TIME = FRAME_LENGTH # constant used in FBE case

is_channel_free = True # free

sim_logger_file = open('logger.txt','w') # logger file with format (time, action, sub_time, node_name))


frames_sent = 0

def csma_ca(env, name, bcs, random_backoff=True):
    global is_channel_free
    global frames_sent
    # arrival time
    arrival_time = random.expovariate(10) # possiion arrival rate every 100 us
    yield env.timeout(arrival_time)
    print('%s:\tarrive \t%s' % (env.now, name))
    sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'arrive', arrival_time, name))

    # maximum NB to access the channel after failure
    NB = 0
    BE = MIN_BACKOFF_TIME
    done = False
    
    
    while NB < MAX_BACKOFFS and not done:
        # backoff periods
        backoff = random.randrange(0, 2**BE , 1) * SLOT_LENGTH
        print('%s:\t%dus backoff \t%s' % (env.now, backoff, name))
        sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'backoff', backoff, name))
        yield env.timeout(backoff)

        # CCA period
        cca = 0
        cca_time = env.now
        while cca < CCA_TIME and is_channel_free == True:
            #print('node %s is_channel_free %d at %s' % (name, is_channel_free, env.now))
            cca += 1
        if cca > 0:
            yield env.timeout(cca)
            print('%s:\t%dus CCA  \t%s' % (cca_time, cca, name))
            sim_logger_file.write('%s\t%s\t%s\t%s\n' % (cca_time, 'CCA', cca, name))
        
        # either send or backoff
        if is_channel_free:
            # send
            is_channel_free = False # channel is occuped
            print('%s:\taccess_channel \t%s' % (env.now, name))
            sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'access_channel', None, name))

            # Frame sending
            frames_sent += 1
            print('%s:\tsend_data \t%s' % (env.now, name))
            sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'send_data', None, name))
            yield env.timeout(FRAME_LENGTH)
            # done
            print('%s:\tfinish_send_data \t%s' % (env.now, name))
            sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'finish_send_data', None, name))
            done = True
            is_channel_free = True

        else: # backoff
            print('%s is increasing BE and NB %s' % (name, env.now))
            NB += 1 
            BE += min(BE + 1, MAX_BACKOFF_TIME)

    if NB > MAX_BACKOFFS:
        print('%s:\tfailed \t%s' % (env.now, name))

env = simpy.Environment()
bcs = simpy.Resource(env, capacity=1)


 
# simulation loop
start_time = env.now

for i in range(FRAMES_TO_SIMULATE):
     env.process(csma_ca(env, 'node_%d' % i, bcs, random_backoff=True))
env.run()

end_time = env.now    

sim_time = end_time - start_time
print('Total simulation time: %s' % sim_time)
print('Transimitted frames: %s success rate:%f' % (frames_sent,frames_sent/FRAMES_TO_SIMULATE))


sim_logger_file.write('Total simulation time: %s\n' % sim_time)
sim_logger_file.write('Transimitted frames: %s success rate:%f\n' % (frames_sent,frames_sent/FRAMES_TO_SIMULATE))


# close logger file
sim_logger_file.close()

