#pragma once

#include <queue>
#include <forward_list>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>

#include "Instruction.hpp"

class ResourceStateQueue {
public:
    ResourceStateQueue(const std::atomic<bool>& shouldStop, long tag)
        : shouldStop(shouldStop), tag(tag), state(MR), pastRChainLength(0)
    {  }

    // represent the state of a resource-state queue
    enum State {N, R, MR};

    void push(std::shared_ptr<Instruction> instruction);
    void listen();
    void notify();
    void startListening();
    void stopListening();
    bool handleNextPendingInstruction();

    // state and transitions
    void toState(ResourceStateQueue::State state);
    void restoreState();

private:
    void priv_toState(ResourceStateQueue::State state);

private:
    std::queue<std::shared_ptr<Instruction>> queue;
    std::mutex queueMutex;
    std::condition_variable queueActivity;
    const std::atomic<bool>& shouldStop;
    const long tag;
    std::mutex stateMutex;
    std::unique_ptr<std::thread> listenThread;
    std::forward_list<std::unique_ptr<std::thread>> workThreads;
    // the current state of the resource-state queue
    ResourceStateQueue::State state;

    // number of R states
    short pastRChainLength;
};
