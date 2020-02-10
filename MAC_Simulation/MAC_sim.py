#!/usr/bin/env python
""" 
Simplified unslotted CSMA/CA algorithm

"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"


import simpy
import random
from collections import deque
from enum import Enum


####### Paramaters #######
# sim params
SIM_TIME = 1000 * 1e6 # 10 seoconds
NUM_NODES = 200
ACK_FAILURE_PROBABILITY = 0.1 # this paramater simulate the environement for ack not being recieved
                              # the higher this value the higher the packet drop rate

# MAC params
SYMBOL_TIME = 16  #us
SLOT_LENGTH = 20 * SYMBOL_TIME    # 20 symbols
FRAME_LENGTH = 10000  # 10 ms(as the paper)    10 * SLOT_LENGTH # 10 slots
#CCA_TIME = 8 * SYMBOL_TIME       # 8 symbols
IDLE_TIME = 0.05 * FRAME_LENGTH       # 5% FRAME_LENGTH
# min and max BE in slots
MIN_BACKOFF_TIME = 1 
MAX_BACKOFF_TIME = 20
# maximum number of backoffs
MAX_BACKOFFS = 4 # before declaring failure
BACKOFF_TIME = FRAME_LENGTH # constant used in FBE case

# NODE params
BUFFER_SIZE = 1000 
MIN_PKGEN_RATE, MAX_PKGEN_RATE = 1e4, 1e5 # us   
MIN_PK_SIZE, MAX_PK_SIZE = (0.1 * FRAME_LENGTH), FRAME_LENGTH


# Global variables
env = simpy.Environment() # environement
is_channel_free = True # free
sim_logger_file = open('logger.txt','w') # logger file with format (time, action, sub_time, node_name))


############################## Classes ##########################################
class PacketType(Enum):
    NORMAL = 1
    ACK = 2
    RTS = 3
    CTS = 4


class DataUnit: # packet, frame, superframe or anything else
    def __init__(self,
                 arrival_time,
                 size, # in useconds
                 source, 
                 destination = None,
                 type:PacketType = PacketType.NORMAL
                 ): 
        self.arrival_time = arrival_time
        self.size = size
        self.type = type
        self.sent = False


class Node:
    def __init__(self,
                 name,
                 joining_time=0):
        self.name = name
        self.NB = 0
        self.BE = MIN_BACKOFF_TIME
        self.CCA_TIME = 8 * SYMBOL_TIME
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.packet_generation_rate = random.randint(MIN_PKGEN_RATE, MAX_PKGEN_RATE)    
        self.joining_time = joining_time

        # statistics
        self.arrived_packets = 0
        self.transimitted_packets = 0
        self.total_arrival_time = 0
        self.throughput = 0
        self.can_process_new_packet = True

        self.run()

    def __str__(self):
        return '(%s)\tarrived:%d\tsent:%d\tfailed:%d\t success rate:%f\tmean_arrival_time:%dus\tthroughput:%d\n' \
               % (self.name, self.arrived_packets, self.transimitted_packets, \
                  self.arrived_packets-self.transimitted_packets, self.transimitted_packets / self.arrived_packets,  \
                  self.total_arrival_time/self.arrived_packets, self.total_arrival_time)

    # callback after packet submission
    def _set_can_generate_new_packet(self, event):
        self.can_process_new_packet = True

    def _update_results(self, event):
        # stats
        self.arrived_packets += 1
        self.total_arrival_time += self.last_packet.arrival_time

        if self.last_packet.sent:
            self.transimitted_packets += 1
            self.throughput += self.last_packet.size
    
    def run(self, event=None):
        if self.can_process_new_packet:
            env.process(self._generate_packet())

    def _generate_packet(self):
        """ Possiion generation rate every packet_generation_rate usecond
        """
        self.can_process_new_packet = False
        self.processed_event = env.event()
        self.processed_event.callbacks.append(self._set_can_generate_new_packet)
        self.processed_event.callbacks.append(self._update_results)
        self.processed_event.callbacks.append(self.run)

        arrival_time = random.expovariate(1. / self.packet_generation_rate) 
        # random size in us
        size = random.randint(MIN_PK_SIZE, MAX_PK_SIZE)
        packet = DataUnit(arrival_time, size, source=self.name)
        self.buffer.append(packet)
        self.last_packet = packet
        ### Packet arrival 
        # wait for the packet until its arrive time triggers
        yield env.timeout(packet.arrival_time)
        env.process(csma_ca(self, packet))

######################## Simplified unslotted CSMA_CA #############################    
def csma_ca(node:Node, packet:DataUnit):
    """
    Actual simplified unslotted CSMA_CA algorithm
    """
    
    global is_channel_free
    
    # reset node BN and BE
    node.NB = 0
    node.BE = MIN_BACKOFF_TIME

    ### Main loop 
    while node.NB < MAX_BACKOFFS and not packet.sent:
        ## Backoff periods
        backoff = random.randrange(0, 2**node.BE , 1) * SLOT_LENGTH
        print('%s:\t%dus backoff \t%s' % (env.now, backoff, node.name))
        sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'backoff', backoff, node.name))
        yield env.timeout(backoff)

        ## CCA period
        cca = 0
        cca_time_start = env.now
        while cca < node.CCA_TIME and is_channel_free == True:
            cca += 1
        if cca > 0:
            yield env.timeout(cca)
            print('%s:\t%dus CCA  \t%s' % (cca_time_start, cca, node.name))
            sim_logger_file.write('%s\t%s\t%s\t%s\n' % (cca_time_start, 'CCA', cca, node.name))

        # detect colisions    
        if cca == node.CCA_TIME and is_channel_free == False: # colision
            print('%s"\tColision\n' % (env.now))
            sim_logger_file.write('%s\tcolision\n' % env.now)

        ## Send or wait period
        if is_channel_free: # carrier sense - channel is free
            # reserve channel
            is_channel_free = False # channel is occuped
            print('%s:\taccess_channel \t%s' % (env.now, node.name))
            sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'access_channel', None, node.name))

            # send
            print('%s:\tsend_data \t%s' % (env.now, node.name))
            sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'send_data', packet.size, node.name))
            yield env.timeout(packet.size)

            # transimission finished, here we can add the ack watiting event
            print('%s:\tfinish_send_data \t%s' % (env.now, node.name))
            sim_logger_file.write('%s\t%s\t%s\t%s\n' % (env.now, 'finish_send_data', None, node.name))
            packet.sent = True
            is_channel_free = True # release channel
        
        else: # channel is busy ... backoff procedure
            print('%s is increasing BE and NB %s' % (node.name, env.now))
            node.NB += 1 
            node.BE += min(node.BE + 1, MAX_BACKOFF_TIME)
    if node.NB > MAX_BACKOFFS:
        print('%s:\tfailed \t%s' % (env.now, node.name))

    # finished processing packet
    node.processed_event.succeed()


def after_math(nodes):
    for node in nodes:
        print(node)
        sim_logger_file.write(node.__str__())
        

def main():
    # simulate
    start_time = env.now

    # create nodes
    nodes = []
    for i in range(NUM_NODES):
        node = Node('node_%d' % i, env.now)
        nodes.append(node)

    # run the simulation
    env.run(SIM_TIME)

    # collect results
    after_math(nodes)

    end_time = env.now    
    sim_time = end_time - start_time
    
    print('Total simulation time: %s' % sim_time)
    
    sim_logger_file.close()

if __name__ is '__main__':
    main()