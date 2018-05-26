#include <memory>
#include <cassert>
#include <iostream>

#include "DependencyEngine.hpp"
#include "Instruction.hpp"

void DependencyEngine::push(callbackType execFunc,
    std::set<long>& readTags,
    std::set<long>& mutateTags) {

    std::set<long> both;
    both.insert(readTags.begin(), readTags.end());
    both.insert(mutateTags.begin(), mutateTags.end());
    int pendingCount = both.size();

    auto instruction = std::make_shared<Instruction>(execFunc,
        readTags, mutateTags, pendingCount);

    for (long readTag : readTags) {
        if (mutateTags.count(readTag) == 0) {
            std::shared_ptr<ResourceStateQueue> tagQueue = this->queues.at(readTag);
            assert(tagQueue != nullptr);
            tagQueue->push(instruction);
        }
    }

    for (long mutateTag : mutateTags) {
        std::shared_ptr<ResourceStateQueue> tagQueue = this->queues.at(mutateTag);
        assert(tagQueue != nullptr);
        tagQueue->push(instruction);
    }
}

long DependencyEngine::newVariable() {
    long tag = this->currentTag++;

    std::shared_ptr<ResourceStateQueue> queue =
        std::make_shared<ResourceStateQueue>(this->shouldStop, tag);

    this->queues[tag] = queue;

    if (!this->shouldStop) {
        queue->startListening();
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
