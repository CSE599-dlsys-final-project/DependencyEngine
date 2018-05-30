#include "Instruction.hpp"
#include "DependencyEngine.hpp"

Instruction::Instruction(callbackType execFunc, void* callbackArgs,
        const std::unordered_set<long>& readTags,
        const std::unordered_set<long>& mutateTags,
        int pendingCount,
        std::unordered_map<long, std::unique_ptr<ResourceStateQueue>>& queues):
    execFunc(execFunc), callbackArgs(callbackArgs),
    readTags(readTags), mutateTags(mutateTags),
    pendingCount(pendingCount), queues(queues)
{ }


bool Instruction::decrementPcAndIsZero() {
    return this->pendingCount.fetch_sub(1) == 1;
}

void Instruction::run() {
    this->execFunc(this->callbackArgs);
    this->restoreStatesAndNotify();
}

void Instruction::restoreStatesAndNotify(){
    std::unordered_set<long> both;
    both.insert(this->readTags.begin(), this->readTags.end());
    both.insert(this->mutateTags.begin(), this->mutateTags.end());
    for (long tag : both) {
        const std::unique_ptr<ResourceStateQueue>& tagQueue = this->queues.at(tag);
        tagQueue->restoreState();
        tagQueue->notify();
    }
}
