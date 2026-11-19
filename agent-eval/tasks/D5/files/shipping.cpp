#include "brackets.h"

static const std::vector<long> kGrams = {0, 500, 2000, 10000};
static const long kFee[] = {499, 899, 1599, 3999};

long parcel_fee(long grams) { return kFee[bracket(kGrams, grams)]; }

long order_fee(const std::vector<long>& parcels) {
    long total = 0;
    for (long g : parcels) total += parcel_fee(g);
    return total;
}
