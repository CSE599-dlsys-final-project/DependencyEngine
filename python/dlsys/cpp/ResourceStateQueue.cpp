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

void ResourceStateQueue::toState(ResourceStateQueue::State state) {
    std::lock_guard<std::mutex> stateLock(this->stateMutex);
    this->priv_toState(state);
}

void ResourceStateQueue::priv_toState(ResourceStateQueue::State state) {
    // retrieve it only once to save time
    auto s = this->state;
    if (s == ResourceStateQueue::N) {
        if (state == ResourceStateQueue::MR) {
            std::cerr << "Invalid state transition" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (s == ResourceStateQueue::R) {
        if (state == ResourceStateQueue::N) {
            std::cerr << "Invalid state transition" << std::endl;
            exit(EXIT_FAILURE);
        } else if (state == ResourceStateQueue::R) {
            this->pastRChainLength++;
        } else { // MR
            if (this->pastRChainLength != 0) {
                std::cerr << "Invalid state transition" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    } else { // MR
        if (state == ResourceStateQueue::R) this->pastRChainLength++;
    }
    // update the state
    this->state = state;
}

void ResourceStateQueue::restoreState() {
  std::lock_guard<std::mutex> stateLock(this->stateMutex);
  auto s = this->state;
  if (s == ResourceStateQueue::MR) {
      std::cerr << "Invalid state restoration" << std::endl;
      exit(EXIT_FAILURE);
  } else if (s == ResourceStateQueue::R) {
      this->pastRChainLength--;
      if (this->pastRChainLength == 0) {
          this->priv_toState(ResourceStateQueue::MR);
      }
  } else { // N
      this->priv_toState(ResourceStateQueue::MR);
  }
}
