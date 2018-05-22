''' A library to track dependency of operations '''
from __future__ import absolute_import

from queue import Queue
from enum import Enum
from threading import Thread, Event, RLock, Lock

class Dependency_Engine(object):
    def __init__(self):
        # This is a mapping of ResourceTag -> ResourceStateQueue
        self.resource_state_queues = {}

        # We need a way to tell all the queues to stop working,
        # if stop_signal.stop is set to true, then all the queue threads
        # will look at its queue: if its empty, finish the thread; if there
        # is still work, finish it first then stop the thread.
        #
        # All of the queues will share the same stop signal, so that
        # when we set the flag to true, all workers are signaled.
        self.stop_signal = StopSignal()

        # This is a pool of running instructions.
        self.running_instruction_thread_pool = []

    def new_variable(self, name = None):
        rtag = ResourceTag(name)

        q = ThreadedResourceStateQueue(self.stop_signal)
        self.resource_state_queues[rtag] = q

        if not self.stop_signal.stop:
            q.start_listening(rtag, self.resource_state_queues)

        return rtag

    def push(self, exec_func, read_tags, mutate_tags):
        # pending count is the number of unique tags
        pending_count = len(set(read_tags + mutate_tags))
        # create instruction based on given parameters
        instruction = Instruction(
            exec_func, read_tags, mutate_tags, pending_count)

        # push instructions into the queue
        # exclusively read
        for tag in read_tags:
            if not tag in mutate_tags:
                assert tag in self.resource_state_queues
                self.resource_state_queues[tag].push(instruction)
        # mutate or read + mutate
        for tag in mutate_tags:
            assert tag in self.resource_state_queues
            self.resource_state_queues[tag].push(instruction)

    # CAUTION: used for demo only, execute the next avaliable
    # instruction for all tags
    def naive_executor(self):
        for tag in self.resource_state_queues:
            self.resource_state_queues[tag].handle_next_pending_instruction \
                (tag, self.resource_state_queues)

    # fire up a thread for each queue to listen for pushes
    def start_threaded_executor(self):
        # tell the queues to not stop and start listening for incoming work
        self.stop_signal.stop = False
        for tag in self.resource_state_queues:
            self.resource_state_queues[tag].start_listening \
                (tag, self.resource_state_queues)

    def stop_threaded_executor(self):
        # tell the queues to stop
        self.stop_signal.stop = True
        # have all the queues finish processing
        for tag in self.resource_state_queues:
            # this will block until this queue's work is done
            self.resource_state_queues[tag].stop_listening()
        # have all the instruction finish processing
        for instruction in self.running_instruction_thread_pool:
            instruction.join()

# the state of a resource queue
class State(Enum):
    N = 0
    R = 1
    MR = 2

class StateWithMemory(object):
    def __init__(self):
        self.state = State.MR
        # count how many consecutive R states are in the transition chain
        self.r_count = 0
        self.lock = RLock()

    def to(self, state):
        self.lock.acquire()
        # State Transition Rules:
        # (1) MR -> R -> R -> ... -> MR -> (1,2)
        # (2) MR -> N -> MR -> (1, 2)
        if self.state == State.N:
            if state != State.MR:
                raise Exception("Invalid state transition")
        elif self.state == State.R:
            if state == State.N:
                raise Exception("Invalid state transition")
            # track length of R chain
            elif state == State.R:
                self.r_count += 1
            else: # MR
                # must be no more Rs left to transit back to MR
                if self.r_count != 0:
                    raise Exception("Invalid state transition")
        else: # self.state == State.MR:
            if state == State.R:
                self.r_count += 1
        self.state = state
        self.lock.release()

    def isIn(self, state):
        return state == self.state

    def restore(self):
        self.lock.acquire()
        if self.state == State.MR:
            raise Exception("Invalid state restoration")
        elif self.state == State.R:
            self.r_count -= 1
            if self.r_count == 0:
                self.to(State.MR)
        else: # self.state == State.N
            self.to(State.MR)
        self.lock.release()

# tells the queues to stop processing
class StopSignal(object):
    def __init__(self, stop = True):
        self.stop = stop

# Stores a lambda function and its dependencies.
class Instruction(Thread):
    def __init__(self, exec_func, read_tags, mutate_tags, pending_counter):
        super(Instruction, self).__init__()
        self.fn = exec_func
        self.pc = pending_counter
        self.m_tags = mutate_tags
        self.r_tags = read_tags
        self.counter_lock = Lock()

    def decrement_pending_counter(self):
        self.counter_lock.acquire()
        self.pc -= 1
        self.counter_lock.release()

    def run(self):
        # runs the lambda function it holds
        self.fn()

# Resource tag represent a variable / object / etc...
# in the dependency engine. Resource tags with the same
# name will be hashed to the same thing.
class ResourceTag(object):
    def __init__(self, name = None):
        if name is not None:
            self.name = name
        else:
            self.name = "ResourceTag " + str(id(self))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name


# Tracks a queue of Instructions and the avaliability of the queue.
# The state of the queue should always reflect the avaliability of the next
# element in the queue.
class ResourceStateQueue(object):
    def __init__(self):
        # the queue is consisted of pending instructions
        self.queue = Queue()
        # integer represent state of the resource:
        #   N state - not ready for read/mutate
        #   R state - only ready for read
        #   MR state - ready for read/mutate
        #self.state = State.MR
        self.state = StateWithMemory()

    # handles the next pending intruction on this queue
    def handle_next_pending_instruction(self, tag, resource_state_queues, pool = None):
        # no pending instruction
        if self.peek() is None:
            return

        ### resolve the next pending instruction
        # mutate or read + mutate
        if tag in self.peek().m_tags:
            # can do stuff
            if self.state.isIn(State.MR):
                # pop
                instruction = self.pop()
                # change state to N
                self.state.to(State.N)
                instruction.decrement_pending_counter()
                if instruction.pc == 0:
                    # calling the non_threaded version
                    if pool is None:
                        instruction.run()
                        # fake callback
                        changed_tags = set(instruction.m_tags + instruction.r_tags)
                        for ctag in changed_tags:
                            resource_state_queues[ctag].state.restore()
                    else:
                        # start the intruction thread and push into the global pool
                        instruction.start()
                        pool.append(instruction)

        # read only
        elif (tag in self.peek().r_tags) \
            and (not tag in self.peek().m_tags):
            # can do stuff
            if self.state.isIn(State.MR) or \
                self.state.isIn(State.R):
                # pop
                instruction = self.pop()
                # change state to N
                self.state.to(State.R)
                instruction.decrement_pending_counter()
                if instruction.pc == 0:
                    # calling the non_threaded version
                    if pool is None:
                        instruction.run()
                        # fake callback
                        changed_tags = set(instruction.m_tags + instruction.r_tags)
                        for ctag in changed_tags:
                            resource_state_queues[ctag].state.restore()
                    else:
                        # start the intruction thread and push into the global pool
                        instruction.start()
                        pool.append(instruction)
        else:
            raise Exception()

    ### queue functions
    # get the next element without popping it
    def peek(self):
        if len(self.queue.queue) == 0:
            return None
        return self.queue.queue[0]

    # push the next instruction into the queue
    def push(self, instruction):
        self.queue.put(instruction)

    # push the next instruction into the queue
    def pop(self):
        return self.queue.get()

    def __repr__(self):
        num_to_state = {State.N:"N", State.R:"R", State.MR:"MR"}
        return "ResourceStateQueue: " + num_to_state[self.state.state];

class ThreadedResourceStateQueue(ResourceStateQueue):
    def __init__(self, stop_signal):
        super(ThreadedResourceStateQueue, self).__init__()
        self.thread = None
        # if stop_signal.stop is set to be true, then stop processing
        self.stop_signal = stop_signal

    # start a thread that handles queue logic
    def start_listening(self, tag, resource_state_queues):
        # already listening
        if self.thread is None:
            #self.running.clear()
            self.thread = Thread(target=self.listen, args=(tag, resource_state_queues))
            self.thread.start()

    # continue to listen for new instructions
    # until the stop_signal is set and all current works are done
    def listen(self, tag, resource_state_queues):
        while(True):
            # check the stop signal and if the queue is done with instructions
            if (self.stop_signal.stop) and (self.peek() is None):
                break
            else:
                # keep going!
                self.handle_next_pending_instruction(tag, resource_state_queues)

    # signals the thread to stop
    # blocks until all works are done
    def stop_listening(self):
        # no thread is running
        if self.thread is None:
            raise Exception("No thread running.")
        # block until work is done
        self.thread.join()
        # clean up
        self.thread = None
