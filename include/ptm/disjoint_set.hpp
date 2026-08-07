#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ptm {

// Control-plane structure for rebuilding candidate consolidation clusters.
// It is intentionally not used as the active source-to-PA mapping because
// Union-Find cannot efficiently split a dissolved consolidation.
class DisjointSet {
public:
    explicit DisjointSet(std::size_t element_count)
        : parent_(element_count), component_size_(element_count, 1) {
        if (element_count == 0 ||
            element_count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("DisjointSet size is outside uint32 range");
        }
        std::iota(parent_.begin(), parent_.end(), std::uint32_t{0});
    }

    [[nodiscard]] std::size_t element_count() const noexcept {
        return parent_.size();
    }

    [[nodiscard]] std::uint32_t find(std::uint32_t element) {
        check(element);
        std::uint32_t root = element;
        while (parent_[root] != root) {
            root = parent_[root];
        }
        while (parent_[element] != element) {
            const auto next = parent_[element];
            parent_[element] = root;
            element = next;
        }
        return root;
    }

    bool unite(std::uint32_t left, std::uint32_t right) {
        auto left_root = find(left);
        auto right_root = find(right);
        if (left_root == right_root) {
            return false;
        }
        if (component_size_[left_root] < component_size_[right_root]) {
            const auto temporary = left_root;
            left_root = right_root;
            right_root = temporary;
        }
        parent_[right_root] = left_root;
        component_size_[left_root] += component_size_[right_root];
        return true;
    }

    [[nodiscard]] bool connected(std::uint32_t left, std::uint32_t right) {
        return find(left) == find(right);
    }

    [[nodiscard]] std::uint32_t component_size(std::uint32_t element) {
        return component_size_[find(element)];
    }

private:
    void check(std::uint32_t element) const {
        if (element >= parent_.size()) {
            throw std::out_of_range("DisjointSet element");
        }
    }

    std::vector<std::uint32_t> parent_;
    std::vector<std::uint32_t> component_size_;
};

}  // namespace ptm
