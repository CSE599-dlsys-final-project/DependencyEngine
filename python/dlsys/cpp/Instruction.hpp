#pragma once

#include <unordered_set>

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

    const callbackType execFunc;
    const void* callbackArgs;
    const std::unordered_set<long> readTags;
    const std::unordered_set<long> mutateTags;
    int pendingCount;
};
