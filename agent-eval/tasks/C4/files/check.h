#pragma once
#include <cstdio>
#include <cstdlib>
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
#define THROWS(e, T) do { bool t = false; try { (void)(e); } catch (const T&) { t = true; } CHECK(t); } while (0)
