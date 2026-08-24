#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int nibble(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: materialize_hex_fixture INPUT.hex OUTPUT\n";
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        std::cerr << "could not open hex fixture\n";
        return 1;
    }
    std::string digits;
    char value = 0;
    while (input.get(value)) {
        if (std::isspace(static_cast<unsigned char>(value)) == 0) {
            digits.push_back(value);
        }
    }
    if (digits.empty() || digits.size() % 2U != 0) {
        std::cerr << "hex fixture has invalid length\n";
        return 1;
    }
    std::vector<std::uint8_t> decoded;
    decoded.reserve(digits.size() / 2U);
    for (std::size_t index = 0; index < digits.size(); index += 2U) {
        const auto high = nibble(digits[index]);
        const auto low = nibble(digits[index + 1U]);
        if (high < 0 || low < 0) {
            std::cerr << "hex fixture contains a non-hex character\n";
            return 1;
        }
        decoded.push_back(static_cast<std::uint8_t>((high << 4) | low));
    }
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(decoded.data()),
        static_cast<std::streamsize>(decoded.size()));
    if (!output) {
        std::cerr << "could not write decoded fixture\n";
        return 1;
    }
    return 0;
}
