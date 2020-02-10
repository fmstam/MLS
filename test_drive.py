

import simpy

def foo(event):
   print('called' + event)

env = simpy.Environment()
event = env.event()
event.callbacks.append(foo)
def st(env):
   yield env.timeout(1)
   event.succeed()
env.process(st(env))
env.run()