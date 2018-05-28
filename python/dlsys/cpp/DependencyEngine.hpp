#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <atomic>

#include "Instruction.hpp"
#include "ResourceStateQueue.hpp"

class DependencyEngine {
public:
    DependencyEngine() : currentTag(0), shouldStop(true)
    { }

    void push(callbackType execFunc, void* callbackArgs,
        const std::unordered_set<long>& readTags,
        const std::unordered_set<long>& mutateTags);

    long newVariable();

    void start();
    void stop();
private:
    std::unordered_map<long, std::unique_ptr<ResourceStateQueue>> queues;
    long currentTag;
    std::atomic<bool> shouldStop;
};
