#pragma once

#include <set>

using callbackType = void(*)(void*);
// (^ a void-returning function that takes a single void pointer argument.)

class Instruction {
public:
    Instruction(callbackType execFunc, void* callbackArgs,
            const std::set<long>& readTags, const std::set<long>& mutateTags,
            int pendingCount):
        execFunc(execFunc), callbackArgs(callbackArgs),
        readTags(readTags), mutateTags(mutateTags),
        pendingCount(pendingCount)
    { }

    callbackType execFunc;
    void* callbackArgs;
    std::set<long> readTags;
    std::set<long> mutateTags;
    int pendingCount;
};
