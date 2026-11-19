#include "plans.h"
#include <cstdio>
#include <fstream>
#include <sstream>

int main() {
    std::ifstream f("config/plans.csv");
    std::string line, plan, limit;
    int rows = 0;
    for (std::getline(f, line); std::getline(f, line); rows++) {
        std::istringstream row(line);
        std::getline(row, plan, ',');
        std::getline(row, limit, ',');
        if (burst_limit(plan) != std::stoi(limit)) { std::printf("FAIL %s\n", plan.c_str()); return 1; }
    }
    std::printf("%s: %d plans\n", rows ? "ok" : "FAIL", rows);
    return rows ? 0 : 1;
}
