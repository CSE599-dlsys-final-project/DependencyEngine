#pragma once

#include <unordered_set>
#include <unordered_map>
#include <iostream>

// forward declare
class ResourceStateQueue;

using callbackType = void(*)(void*);
// (^ a void-returning function that takes a single void pointer argument.)

class Instruction {
public:
    Instruction(callbackType execFunc, void* callbackArgs,
            const std::unordered_set<long>& readTags,
            const std::unordered_set<long>& mutateTags,
            int pendingCount,
            std::unordered_map<long, std::unique_ptr<ResourceStateQueue>>& queues);
    bool decrementPcAndIsZero();
    void run();
    void restoreStatesAndNotify();

    const callbackType execFunc;
    void* callbackArgs;
    const std::unordered_set<long> readTags;
    const std::unordered_set<long> mutateTags;
    std::atomic<int> pendingCount;
    std::unordered_map<long, std::unique_ptr<ResourceStateQueue>>& queues;
};
