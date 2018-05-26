#include "ResourceStateQueue.hpp"

void ResourceStateQueue::push(std::shared_ptr<Instruction> instruction) {
    {
        std::lock_guard<std::mutex> lock(this->queueMutex);
        this->queue.push(instruction);
    }

    this->queueActivity.notify_one();
}

void ResourceStateQueue::listen() {
    // Performs the work of reading from this queue and dispatching instructions.

    while (true) {
        std::unique_lock<std::mutex> queueLock(this->queueMutex);

        this->queueActivity.wait(queueLock,
            [this] { return !this->queue.empty() || this->shouldStop; });

        if (this->shouldStop && this->queue.empty()) {
            return;
        }

        while (this->handleNextPendingInstruction()) {
            // (keep going...)
        }
    }
}

void ResourceStateQueue::notify() {
    this->queueActivity.notify_one();
}

void ResourceStateQueue::startListening() {

}

void ResourceStateQueue::stopListening() {

}

bool ResourceStateQueue::handleNextPendingInstruction() {
    return false;
}
