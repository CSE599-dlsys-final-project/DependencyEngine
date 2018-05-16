''' A library to track dependency of operations '''
from __future__ import absolute_import

from queue import Queue

class Dependency_Engine(object):
    def __init__(self):
        self.resource_state_queues = {}

    def new_variable(self, name = None):
        rtag = ResourceTag(name)
        self.resource_state_queues[rtag] = ResourceStateQueue()
        return rtag

    def push(self, exec_func, read_tags, mutate_tags):
        # pending count is the number of unique tags
        pending_count = len(set(read_tags + mutate_tags))
        # create instruction based on given parameters
        instruction = Instruction(exec_func, read_tags, mutate_tags, pending_count)

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

    # CAUTION: used for demo only, execute the next avaliable instruction for all tags
    def fake_executor(self):
        for tag in self.resource_state_queues:
            # no instructions pending
            if self.resource_state_queues[tag].peek() == None:
                continue
            ### resolve the next pending instruction
            # mutate or read + mutate
            if tag in self.resource_state_queues[tag].peek().m_tags:
                # can do stuff
                if self.resource_state_queues[tag].state == 2:
                    # pop
                    instruction = self.resource_state_queues[tag].pop()
                    # change state to N
                    self.resource_state_queues[tag].state = 0
                    instruction.pc = instruction.pc - 1
                    if instruction.pc == 0:
                        instruction.fn()
                        # fake callback
                        for changed_tag in instruction.m_tags:
                            self.resource_state_queues[changed_tag].state = 2
                        for changed_tag in instruction.r_tags:
                            self.resource_state_queues[changed_tag].state = 2
            # read only
            elif (tag in self.resource_state_queues[tag].peek().r_tags) \
                and (not tag in self.resource_state_queues[tag].peek().m_tags):
                # can do stuff
                if self.resource_state_queues[tag].state == 2 or \
                    self.resource_state_queues[tag].state == 1:
                    prev = self.resource_state_queues[tag].state
                    # pop
                    instruction = self.resource_state_queues[tag].pop()
                    # change state to N
                    self.resource_state_queues[tag].state = 1
                    instruction.pc = instruction.pc - 1
                    if instruction.pc == 0:
                        instruction.fn()
                        # fake callback
                        for changed_tag in instruction.m_tags:
                            self.resource_state_queues[changed_tag].state = 2
                        for changed_tag in instruction.r_tags:
                            self.resource_state_queues[changed_tag].state = prev

class ResourceTag(object):
    def __init__(self, name = None):
        if name != None:
            self.name = name
        else:
            self.name = "ResourceTag"

    def __repr__(self):
        return self.name

class ResourceStateQueue(object):
    def __init__(self):
        # the queue is consisted of pending instructions
        self.queue = Queue()
        # integer represent state of the resource:
        #   0 (N state) - not ready for read/mutate
        #   1 (R state) - only ready for read
        #   2 (MR state) - ready for read/mutate
        self.state = 2

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
        num_to_state = {0:"N", 1:"R", 2:"MR"}
        return "ResourceStateQueue: " + num_to_state[self.state];

class Instruction(object):
    def __init__(self, exec_func, read_tags, mutate_tags, pending_counter):
        self.fn = exec_func
        self.pc = pending_counter
        self.m_tags = mutate_tags
        self.r_tags = read_tags
