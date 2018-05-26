#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

#include "Instruction.hpp"

class ResourceStateQueue {
public:
    void push(std::shared_ptr<Instruction> instruction) {
        this->queue.push(instruction);
    }

private:
    std::queue<std::shared_ptr<Instruction>> queue;
    std::condition_variable cv;
};
