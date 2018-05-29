#pragma once

#include <unordered_set>
#include <iostream>

using callbackType = void(*)(void*);
// (^ a void-returning function that takes a single void pointer argument.)

class Instruction {
public:
    Instruction(callbackType execFunc, void* callbackArgs,
            const std::unordered_set<long>& readTags,
            const std::unordered_set<long>& mutateTags,
            int pendingCount):
        execFunc(execFunc), callbackArgs(callbackArgs),
        readTags(readTags), mutateTags(mutateTags),
        pendingCount(pendingCount)
    { }

    bool decrementPcAndIsZero() {
        return this->pendingCount.fetch_sub(1) == 1;
    }

    void run() {
        std::cerr << "Calling callback..." << std::endl;
        this->execFunc(this->callbackArgs);
        std::cerr << "Done." << std::endl;
        //this->restoreStates !! TODO
    }

    const callbackType execFunc;
    void* callbackArgs;
    const std::unordered_set<long> readTags;
    const std::unordered_set<long> mutateTags;
    std::atomic<int> pendingCount;
};
