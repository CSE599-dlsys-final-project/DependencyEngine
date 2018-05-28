#include <memory>

#include "DependencyEngine.hpp"
#include "Instruction.hpp"

void DependencyEngine::push(callbackType execFunc, void* callbackArgs,
    const std::unordered_set<long>& readTags,
    const std::unordered_set<long>& mutateTags) {

    std::unordered_set<long> both;
    both.insert(readTags.begin(), readTags.end());
    both.insert(mutateTags.begin(), mutateTags.end());
    int pendingCount = both.size();

    auto instruction = std::make_shared<Instruction>(execFunc, callbackArgs,
        readTags, mutateTags, pendingCount);

    // As a test, we can execute the thing right here:
    // execFunc(callbackArgs);

    for (long readTag : readTags) {
        if (mutateTags.count(readTag) == 0) {
            std::unique_ptr<ResourceStateQueue>& tagQueue = this->queues.at(readTag);
            tagQueue->push(instruction);
        }
    }

    for (long mutateTag : mutateTags) {
        std::unique_ptr<ResourceStateQueue>& tagQueue = this->queues.at(mutateTag);
        tagQueue->push(instruction);
    }
}

long DependencyEngine::newVariable() {
    long tag = this->currentTag++;

    this->queues[tag] = std::make_unique<ResourceStateQueue>(this->shouldStop, tag);

    if (!this->shouldStop) {
        this->queues[tag]->startListening();
    }

    return tag;
}

void DependencyEngine::start() {
    this->shouldStop = false;

    for (const auto& entry : this->queues) {
        entry.second->startListening();
    }
}

void DependencyEngine::stop() {
    this->shouldStop = true;

    // TODO: stop listening on each ResourceStateQueue, join on
    // all threads.
}
