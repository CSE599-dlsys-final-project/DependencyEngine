#pragma once

#include <string>
#include <unordered_map>
#include <set>
#include <memory>

#include "Instruction.hpp"
#include "ResourceStateQueue.hpp"

class DependencyEngine {
public:
    DependencyEngine() : currentTag(0), shouldStop(true)
    { }

    void push(callbackType execFunc,
        std::set<long>& readTags,
        std::set<long>& mutateTags);

    long newVariable();

    void start();
    void stop();
private:
    std::unordered_map<long, std::shared_ptr<ResourceStateQueue>> queues;
    long currentTag;
    bool shouldStop;
};
