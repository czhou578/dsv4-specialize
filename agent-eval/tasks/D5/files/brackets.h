#pragma once
#include <vector>

// The index of the last threshold that is <= x (thresholds ascending, thr[0] = 0).
int bracket(const std::vector<long>& thr, long x);
long parcel_fee(long grams);                       // in cents
long order_fee(const std::vector<long>& parcels);  // in cents
