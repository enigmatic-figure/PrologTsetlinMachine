#include "ptm/bit_block.hpp"

namespace ptm {

const char* port_semantic_name(PortSemantic semantic) noexcept {
    switch (semantic) {
        case PortSemantic::literal_truth:
            return "literal_truth";
        case PortSemantic::ta_action:
            return "ta_action";
        case PortSemantic::literal_condition:
            return "literal_condition";
        case PortSemantic::clause_output:
            return "clause_output";
    }
    return "unknown";
}

}  // namespace ptm

