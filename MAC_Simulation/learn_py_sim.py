# a learning class of simpy
import simpy
import random

FRAME_LENGTH = 1000 # 10 ms
CCA_TIME = 20        # 20 us
IDLE_TIME = 50       # 5% FRAME_LENGTH
BACKOFF_TIME = FRAME_LENGTH # constant
MIN_BACKOFF_TIME = 15
MAX_BACKOFF_TIME = 1023
MAX_ATTEMPTS = 7 # before declaring failure
is_channel_free = True # free

sim_logger_file = open('logger.txt','w') # logger file with format (time, action, sub_time, node_name))


frames_sent = 0

def node(env, name, bcs, random_backoff=True):
    global is_channel_free
    global frames_sent
    # arrival time
    arrival_time = random.expovariate(1. / 50.) # possiion arrival rate every 10 s
    yield env.timeout(arrival_time)
    print('%s:\tarrive \t%s' % (env.now, name))
    sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'arrive', None, name))

    # maximum attempts to access the channel after failure
    attempts = 0
    done = False
    
    with bcs.request() as req:
        while attempts < MAX_ATTEMPTS and not done:
            print('%s:\tch_access_attempt(%d) \t%s' %(env.now, attempts, name))
            sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'ch_access_attempt', None, name))

            # CCA period
            cca = 0
            cca_time = env.now
            while cca < CCA_TIME and is_channel_free == True:
                #print('node %s is_channel_free %d at %s' % (name, is_channel_free, env.now))
                yield env.timeout(1)
                cca += 1
            if cca > 0:
                print('%s:\t%dus CCA \t%s' % (cca_time, cca, name))
                sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'CCA', cca_time, name))
            
            # send or backoff decision period
            res = yield req | env.timeout(0)
            
            if req in res:
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
                attempts += 1  
                if random_backoff:
                    #backoff = random.randrange(MIN_BACKOFF_TIME, MAX_BACKOFF_TIME, 1) * 1000 
                    backoff = random.randrange(0, 2**attempts * 1000 , 1) 
                else:
                    backoff = BACKOFF_TIME

                print('%s:\t%dus backoff \t%s' % (env.now, backoff, name))
                sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'backoff', backoff, name))
                yield env.timeout(backoff)

    if attempts >= MAX_ATTEMPTS:
        print('%s:\tfailed \t%s' % (env.now, name))


env = simpy.Environment()
bcs = simpy.Resource(env, capacity=1)

# simulation loop
start_time = env.now
frames_to_send = 10
for i in range(frames_to_send):
     env.process(node(env, 'node_%d' % i, bcs, random_backoff=True))
env.run()

end_time = env.now    

sim_time = end_time - start_time
print('Total simulation time: %s' % sim_time)
print('Transimitted frames: %s success rate:%f' % (frames_sent,frames_sent/frames_to_send))


sim_logger_file.write('Total simulation time: %s\n' % sim_time)
sim_logger_file.write('Transimitted frames: %s success rate:%f\n' % (frames_sent,frames_sent/frames_to_send))


# close logger file
sim_logger_file.close()

