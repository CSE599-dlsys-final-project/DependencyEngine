#pragma once

#include <set>

using callbackType = long; // Revisit this.

class Instruction {
public:
    Instruction(callbackType execFunc,
        std::set<long> readTags,
        std::set<long> mutateTags, int pendingCount):
        execFunc(execFunc), readTags(readTags), mutateTags(mutateTags),
        pendingCount(pendingCount)
    { }

    callbackType execFunc;
    std::set<long> readTags;
    std::set<long> mutateTags;
    int pendingCount;
};
