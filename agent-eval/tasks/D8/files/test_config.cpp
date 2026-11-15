#include <cstdio>
#include <fstream>
#include <sstream>
#include "config.h"

static std::string slurp(const char* path) {
    std::ifstream f(path);
    if (!f) { std::printf("cannot open %s\n", path); return "\x01"; }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main() {
    std::string got, expected = slurp("data/expected.txt");
    for (auto& [k, v] : parse_config(slurp("data/sample.conf"))) got += k + "=" + v + "\n";
    if (got != expected) { std::printf("FAIL: got\n%s", got.c_str()); return 1; }
    std::printf("all checks passed\n");
    return 0;
}
