#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

#include "Instruction.hpp"

class ResourceStateQueue {
public:
    ResourceStateQueue(const std::atomic<bool>& shouldStop, long tag)
        : shouldStop(shouldStop), tag(tag)
    { }

    void push(std::shared_ptr<Instruction> instruction);
    void listen();
    void notify();
    void startListening();
    void stopListening();
    bool handleNextPendingInstruction();

private:
    std::queue<std::shared_ptr<Instruction>> queue;
    std::mutex queueMutex;
    std::condition_variable queueActivity;
    const std::atomic<bool>& shouldStop;
    const long tag;
};
