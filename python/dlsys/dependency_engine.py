''' A library to track dependency of operations '''
from __future__ import absolute_import

import queue
from enum import Enum
from threading import Thread, Event, RLock, Lock, Condition
from contextlib import contextmanager

class DependencyEngine(object):
    def __init__(self, concurrent_instructions = True):
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
        if concurrent_instructions:
            self.running_instruction_thread_pool = []
        else:
            self.running_instruction_thread_pool = None

    def new_variable(self, name = None):
        rtag = ResourceTag(name)

        q = ThreadedResourceStateQueue(self.stop_signal,
            self.running_instruction_thread_pool)
        self.resource_state_queues[rtag] = q

        if not self.stop_signal.stop:
            q.start_listening(rtag, self.resource_state_queues)

        return rtag

    def push(self, exec_func, read_tags, mutate_tags):
        # pending count is the number of unique tags
        pending_count = len(set(read_tags + mutate_tags))
        # create instruction based on given parameters
        instruction = Instruction(
            exec_func, read_tags, mutate_tags, pending_count,
            self.resource_state_queues)

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

    @contextmanager
    def threaded_executor(self):
        self.start_threaded_executor()
        yield
        self.stop_threaded_executor()

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
        # State Transition Rules:
        # (1) MR -> R -> R -> ... -> MR -> (1,2)
        # (2) MR -> N -> MR -> (1, 2)
        with self.lock:
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

    def isIn(self, state):
        return state == self.state

    def restore(self):
        with self.lock:
            if self.state == State.MR:
                #print self.history
                raise Exception("Invalid state restoration")
            elif self.state == State.R:
                self.r_count -= 1
                if self.r_count == 0:
                    self.to(State.MR)
            else: # self.state == State.N
                self.to(State.MR)

# tells the queues to stop processing
class StopSignal(object):
    def __init__(self, stop = True):
        self.stop = stop

# Stores a lambda function and its dependencies.
class Instruction(Thread):
    def __init__(self, exec_func, read_tags, mutate_tags, pending_counter,
                resource_state_queues):
        super(Instruction, self).__init__()
        self.fn = exec_func
        self.pc = pending_counter
        self.m_tags = mutate_tags
        self.r_tags = read_tags
        self.resource_state_queues = resource_state_queues
        self.counter_lock = Lock()

    # decrement the pc counter and returns true if
    # the counter is zero or false otherwise
    def decrement_pc_and_is_zero(self):
        with self.counter_lock:
            self.pc -= 1
            return self.pc == 0

    # runs the lambda function it holds
    def run(self):
        self.fn()
        self.restore_states()

    # restore the states that was changed previously
    # by the resource state queue
    def restore_states(self):
        changed_tags = set(self.m_tags + self.r_tags)
        for ctag in changed_tags:
            self.resource_state_queues[ctag].state.restore()
            self.resource_state_queues[ctag].notify()

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
        self.queue = queue.Queue()
        # integer represent state of the resource:
        #   N state - not ready for read/mutate
        #   R state - only ready for read
        #   MR state - ready for read/mutate
        #self.state = State.MR
        self.state = StateWithMemory()

        # Used for either new item on queue, or needing to stop.
        self.queueActivity = Condition()

    def handle_next_pending_instruction(self, tag, resource_state_queues, pool = None):
        """
        handles the next pending intruction on this queue.
        Returns whether it popped and handled an instruction.
        """
        # no pending instruction
        if self.peek() is None:
            return False

        ### resolve the next pending instruction
        # mutate or read + mutate
        if tag in self.peek().m_tags:
            # can do stuff
            if self.state.isIn(State.MR):
                # change state to N
                self.state.to(State.N)
                instruction = self.pop()

                if instruction.decrement_pc_and_is_zero():
                    # calling the non_threaded version
                    if pool is None:
                        instruction.run()
                    else:
                        # start the intruction thread and push into the global pool
                        instruction.start()
                        pool.append(instruction)

                return True

        # read only
        elif (tag in self.peek().r_tags) \
            and (not tag in self.peek().m_tags):
            # can do stuff
            if self.state.isIn(State.MR) or \
                self.state.isIn(State.R):
                # change state to N
                self.state.to(State.R)
                instruction = self.pop()

                if instruction.decrement_pc_and_is_zero():
                    # calling the non_threaded version
                    if pool is None:
                        instruction.run()
                    else:
                        # start the intruction thread and push into the global pool
                        instruction.start()
                        pool.append(instruction)

                return True
        else:
            raise Exception()

        return False

    # wakes the queue up
    def notify(self):
        with self.queueActivity:
            self.queueActivity.notify()

    ### queue functions
    # get the next element without popping it
    def peek(self):
        if len(self.queue.queue) == 0:
            return None
        return self.queue.queue[0]

    # push the next instruction into the queue
    def push(self, instruction):
        with self.queueActivity:
            self.queue.put(instruction)
            self.queueActivity.notify()

    # push the next instruction into the queue
    def pop(self):
        return self.queue.get()

    def __repr__(self):
        num_to_state = {State.N:"N", State.R:"R", State.MR:"MR"}
        return "ResourceStateQueue: " + num_to_state[self.state.state];

class ThreadedResourceStateQueue(ResourceStateQueue):
    def __init__(self, stop_signal, intruction_thread_pool = None):
        super(ThreadedResourceStateQueue, self).__init__()
        self.thread = None
        # if stop_signal.stop is set to be true, then stop processing
        self.stop_signal = stop_signal
        # the pool of running instruction threads
        self.pool = intruction_thread_pool

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
        def handle_instruction():
            if self.handle_next_pending_instruction(tag, resource_state_queues,
                                                    self.pool):
                self.queue.task_done()
                return True
            else:
                return False

        while True:
            should_wake = lambda: (not self.queue.empty()
                or self.stop_signal.stop)

            with self.queueActivity:
                self.queueActivity.wait_for(should_wake)

                if self.stop_signal.stop and self.queue.empty():
                    return
                else:
                    # Service single item and continue.
                    # handle consecutive reads
                    while handle_instruction():
                        pass

    # signals the thread to stop
    # blocks until all works are done
    def stop_listening(self):
        # The caller (probably) changed the stop signal. Notify our waiting thread.
        with self.queueActivity:
            self.queueActivity.notify()

        # no thread is running
        if self.thread is None:
            raise Exception("No thread running.")
        # block until work is done
        self.thread.join()
        # clean up
        self.thread = None
