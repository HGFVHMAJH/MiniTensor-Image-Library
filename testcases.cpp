#include "Tensor.hpp"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace std;

int passed = 0;
int failed = 0;
int notSupported = 0;

void pass(const string& id)
{
    cout << id << " : PASS\n";
    ++passed;
}

void fail(const string& id, const string& message)
{
    cout << id << " : FAIL";
    if (!message.empty())
        cout << " - " << message;
    cout << '\n';
    ++failed;
}

void unsupported(const string& id)
{
    cout << id << " : NOT_SUPPORTED\n";
    ++notSupported;
}

bool near(float actual, float expected)
{
    return fabs(actual - expected) <=
           1e-5f + 1e-4f * fabs(expected);
}

bool equal(const vector<float>& actual,
           const vector<float>& expected)
{
    if (actual.size() != expected.size())
        return false;

    for (size_t i = 0; i < actual.size(); ++i)
        if (!near(actual[i], expected[i]))
            return false;

    return true;
}

int main()
{
    using T = Tensor<float>;

    // TC01: Scalar tensor
    try {
        T t(vector<size_t>{}, vector<float>{5});
        if (t.rank() == 0 && t.size() == 1 &&
            t.shape().empty() && near(t.data()[0], 5))
            pass("TC01");
        else
            fail("TC01", "Incorrect scalar properties");
    }
    catch (const exception& e) { fail("TC01", e.what()); }

    // TC02: 1D tensor
    try {
        T t({5}, {1, 2, 3, 4, 5});
        if (t.shape() == vector<size_t>{5} &&
            equal(t.data(), {1, 2, 3, 4, 5}))
            pass("TC02");
        else
            fail("TC02", "Incorrect 1D tensor");
    }
    catch (const exception& e) { fail("TC02", e.what()); }

    // TC03: 2D tensor
    try {
        T t({2, 3}, {1, 2, 3, 4, 5, 6});
        if (t.shape() == vector<size_t>{2, 3} &&
            equal(t.data(), {1, 2, 3, 4, 5, 6}))
            pass("TC03");
        else
            fail("TC03", "Incorrect 2D tensor");
    }
    catch (const exception& e) { fail("TC03", e.what()); }

    // TC04: 3D tensor
    try {
        T t({2, 2, 3}, vector<float>(12));
        if (t.rank() == 3 && t.size() == 12)
            pass("TC04");
        else
            fail("TC04", "Incorrect 3D tensor");
    }
    catch (const exception& e) { fail("TC04", e.what()); }

    // TC05: 4D tensor
    try {
        T t({2, 3, 2, 2}, vector<float>(24));
        if (t.rank() == 4 && t.size() == 24)
            pass("TC05");
        else
            fail("TC05", "Incorrect 4D tensor");
    }
    catch (const exception& e) { fail("TC05", e.what()); }

    // TC06: Default initialization
    try {
        T t({2, 3});
        bool okay = true;
        for (float x : t.data())
            okay &= (x == 0);

        if (okay) pass("TC06");
        else fail("TC06", "Elements are not initialized to zero");
    }
    catch (const exception& e) { fail("TC06", e.what()); }

    // TC07: Contiguous storage
    try {
        T t({2, 3}, {1, 2, 3, 4, 5, 6});
        const float* p = t.data().data();
        bool okay = true;

        for (size_t i = 0; i < t.size() - 1; ++i)
            okay &= (&p[i + 1] - &p[i]) == 1;

        if (okay && sizeof(float) == 4) pass("TC07");
        else fail("TC07", "Storage is not contiguous float32");
    }
    catch (const exception& e) { fail("TC07", e.what()); }

    // TC08: Large allocation
    try {
        T t({1000, 1000});
        t.fill(2.0f);

        double sum = 0;
        for (float x : t.data()) sum += x;

        if (t.size() == 1000000 && sum == 2000000.0)
            pass("TC08");
        else
            fail("TC08", "Incorrect size or sum");
    }
    catch (const exception& e) { fail("TC08", e.what()); }

    // TC09: Read (0,0)
    try {
        T t({2, 3}, {1, 2, 3, 4, 5, 6});
        if (t.getElement({0, 0}) == 1) pass("TC09");
        else fail("TC09", "Incorrect value");
    }
    catch (const exception& e) { fail("TC09", e.what()); }

    // TC10: Read (1,2)
    try {
        T t({2, 3}, {1, 2, 3, 4, 5, 6});
        if (t.getElement({1, 2}) == 6) pass("TC10");
        else fail("TC10", "Incorrect value");
    }
    catch (const exception& e) { fail("TC10", e.what()); }

    // TC11: Read (1,1)
    try {
        T t({2, 3}, {1, 2, 3, 4, 5, 6});
        if (t.getElement({1, 1}) == 5) pass("TC11");
        else fail("TC11", "Incorrect value");
    }
    catch (const exception& e) { fail("TC11", e.what()); }

    // TC12: Set element
    try {
        T t({2, 3}, {1, 2, 3, 4, 5, 6});
        t.setElement(20, {0, 1});

        if (equal(t.data(), {1, 20, 3, 4, 5, 6}))
            pass("TC12");
        else
            fail("TC12", "Mutation incorrect");
    }
    catch (const exception& e) { fail("TC12", e.what()); }

    // TC13: Invalid row
    try {
        T t({2, 3});
        bool threw = false;
        try { (void)t.getElement({2, 0}); }
        catch (const out_of_range&) { threw = true; }

        if (threw) pass("TC13");
        else fail("TC13", "Expected out_of_range");
    }
    catch (const exception& e) { fail("TC13", e.what()); }

    // TC14: Invalid column
    try {
        T t({2, 3});
        bool threw = false;
        try { (void)t.getElement({0, 3}); }
        catch (const out_of_range&) { threw = true; }

        if (threw) pass("TC14");
        else fail("TC14", "Expected out_of_range");
    }
    catch (const exception& e) { fail("TC14", e.what()); }

    // TC15: Negative index
    unsupported("TC15");

    // TC16: 3D index mapping
    try {
        vector<float> values(24);
        for (size_t i = 0; i < 24; ++i)
            values[i] = static_cast<float>(i);

        T t({2, 3, 4}, values);

        if (t.getElement({1, 2, 3}) == 23)
            pass("TC16");
        else
            fail("TC16", "Incorrect flattened index");
    }
    catch (const exception& e) { fail("TC16", e.what()); }

    // TC17: Addition
    try {
        T a({2, 2}, {1, 2, 3, 4});
        T b({2, 2}, {5, 6, 7, 8});

        if (equal((a + b).data(), {6, 8, 10, 12}))
            pass("TC17");
        else
            fail("TC17", "Incorrect addition");
    }
    catch (const exception& e) { fail("TC17", e.what()); }

    // TC18: Subtraction
    try {
        T a({2, 2}, {1, 2, 3, 4});
        T b({2, 2}, {5, 6, 7, 8});

        if (equal((a - b).data(), {-4, -4, -4, -4}))
            pass("TC18");
        else
            fail("TC18", "Incorrect subtraction");
    }
    catch (const exception& e) { fail("TC18", e.what()); }

    // TC19: Addition by zero
    try {
        T a({2, 2}, {1, 2, 3, 4});
        T b({2, 2});

        if (equal((a + b).data(), a.data())) pass("TC19");
        else fail("TC19", "Addition by zero changed values");
    }
    catch (const exception& e) { fail("TC19", e.what()); }

    // TC20: Subtraction of zero
    try {
        T a({2, 2}, {1, 2, 3, 4});
        T b({2, 2});

        if (equal((a - b).data(), a.data())) pass("TC20");
        else fail("TC20", "Subtraction of zero changed values");
    }
    catch (const exception& e) { fail("TC20", e.what()); }

    // TC21: Signed addition
    try {
        T a({2, 2}, {-2, .5f, 3, -4});
        T b({2, 2}, {1, -.5f, -3, 2});

        if (equal((a + b).data(), {-1, 0, 0, -2}))
            pass("TC21");
        else fail("TC21", "Incorrect signed addition");
    }
    catch (const exception& e) { fail("TC21", e.what()); }

    // TC22: Signed subtraction
    try {
        T a({2, 2}, {-2, .5f, 3, -4});
        T b({2, 2}, {1, -.5f, -3, 2});

        if (equal((a - b).data(), {-3, 1, 6, -6}))
            pass("TC22");
        else fail("TC22", "Incorrect signed subtraction");
    }
    catch (const exception& e) { fail("TC22", e.what()); }

    // TC23: Addition shape mismatch
    try {
        T a({2, 3}), b({3, 2});
        bool threw = false;
        try { (void)(a + b); }
        catch (const invalid_argument&) { threw = true; }

        if (threw) pass("TC23");
        else fail("TC23", "Expected invalid_argument");
    }
    catch (const exception& e) { fail("TC23", e.what()); }

    // TC24: Subtraction shape mismatch
    try {
        T a({2, 3}), b({6});
        bool threw = false;
        try { (void)(a - b); }
        catch (const invalid_argument&) { threw = true; }

        if (threw) pass("TC24");
        else fail("TC24", "Expected invalid_argument");
    }
    catch (const exception& e) { fail("TC24", e.what()); }

    // TC25: Element-wise multiplication
    try {
        T a({2, 2}, {1, 2, 3, 4});
        T b({2, 2}, {5, 6, 7, 8});

        if (equal((a * b).data(), {5, 12, 21, 32}))
            pass("TC25");
        else fail("TC25", "Incorrect multiplication");
    }
    catch (const exception& e) { fail("TC25", e.what()); }

    // TC26: Multiplication by zero
    try {
        T a({2, 2}, {1, 2, 3, 4}), b({2, 2});

        if (equal((a * b).data(), {0, 0, 0, 0}))
            pass("TC26");
        else fail("TC26", "Expected zero result");
    }
    catch (const exception& e) { fail("TC26", e.what()); }

    // TC27: Multiplication by one
    try {
        T a({2, 2}, {1, 2, 3, 4}), b({2, 2});
        b.fill(1);

        if (equal((a * b).data(), a.data())) pass("TC27");
        else fail("TC27", "Multiplication by one changed values");
    }
    catch (const exception& e) { fail("TC27", e.what()); }

    // TC28: Signed multiplication
    try {
        T a({2, 2}, {-2, .5f, 3, -4});
        T b({2, 2}, {1, -.5f, -3, 2});

        if (equal((a * b).data(), {-2, -.25f, -9, -8}))
            pass("TC28");
        else fail("TC28", "Incorrect signed multiplication");
    }
    catch (const exception& e) { fail("TC28", e.what()); }

    // TC29: Fractional multiplication
    try {
        T a({2, 2}, {.1f, .2f, .3f, .4f});
        T b({2, 2}, {.2f, .3f, .4f, .5f});

        if (equal((a * b).data(), {.02f, .06f, .12f, .20f}))
            pass("TC29");
        else fail("TC29", "Incorrect fractional multiplication");
    }
    catch (const exception& e) { fail("TC29", e.what()); }

    // TC30: Multiplication shape mismatch
    try {
        T a({2, 3}), b({3, 2});
        bool threw = false;
        try { (void)(a * b); }
        catch (const invalid_argument&) { threw = true; }

        if (threw) pass("TC30");
        else fail("TC30", "Expected invalid_argument");
    }
    catch (const exception& e) { fail("TC30", e.what()); }

    // TC31-TC38: Matrix multiplication not implemented
    for (int i = 31; i <= 38; ++i)
        unsupported("TC" + to_string(i));

    // TC39: Reshape 1D to 3x4
    try {
        vector<float> values(12);
        for (size_t i = 0; i < 12; ++i) values[i] = float(i);

        T t({12}, values);
        t.reshape({3, 4});

        if (t.shape() == vector<size_t>{3, 4} &&
            equal(t.data(), values)) pass("TC39");
        else fail("TC39", "Incorrect reshape");
    }
    catch (const exception& e) { fail("TC39", e.what()); }

    // TC40: Reshape 3x4 to 2x2x3
    try {
        vector<float> values(12);
        for (size_t i = 0; i < 12; ++i) values[i] = float(i);

        T t({3, 4}, values);
        t.reshape({2, 2, 3});

        if (t.shape() == vector<size_t>{2, 2, 3} &&
            equal(t.data(), values)) pass("TC40");
        else fail("TC40", "Incorrect reshape");
    }
    catch (const exception& e) { fail("TC40", e.what()); }

    // TC41: Reshape 2x2x3 to 1D
    try {
        vector<float> values(12);
        for (size_t i = 0; i < 12; ++i) values[i] = float(i);

        T t({2, 2, 3}, values);
        t.reshape({12});

        if (t.shape() == vector<size_t>{12} &&
            equal(t.data(), values)) pass("TC41");
        else fail("TC41", "Incorrect reshape");
    }
    catch (const exception& e) { fail("TC41", e.what()); }

    // TC42: Reshape 2x3 to 3x2
    try {
        T t({2, 3}, {1, 2, 3, 4, 5, 6});
        t.reshape({3, 2});

        if (t.shape() == vector<size_t>{3, 2} &&
            equal(t.data(), {1, 2, 3, 4, 5, 6}))
            pass("TC42");
        else fail("TC42", "Incorrect reshape");
    }
    catch (const exception& e) { fail("TC42", e.what()); }

    // TC43: Invalid reshape
    try {
        T t({2, 3}, {1, 2, 3, 4, 5, 6});
        bool threw = false;

        try { t.reshape({4, 4}); }
        catch (const invalid_argument&) { threw = true; }

        if (threw &&
            t.shape() == vector<size_t>{2, 3} &&
            equal(t.data(), {1, 2, 3, 4, 5, 6}))
            pass("TC43");
        else fail("TC43", "Invalid reshape not handled correctly");
    }
    catch (const exception& e) { fail("TC43", e.what()); }

    // TC44: Singleton dimensions
    try {
        vector<float> values(12);
        for (size_t i = 0; i < 12; ++i) values[i] = float(i);

        T t({12}, values);
        t.reshape({1, 12, 1});

        if (t.shape() == vector<size_t>{1, 12, 1} &&
            equal(t.data(), values)) pass("TC44");
        else fail("TC44", "Incorrect singleton reshape");
    }
    catch (const exception& e) { fail("TC44", e.what()); }

    // TC45: 100 reshape round trips
    try {
        vector<float> values(12);
        for (size_t i = 0; i < 12; ++i) values[i] = float(i);

        T t({12}, values);
        bool okay = true;

        for (int i = 0; i < 100; ++i) {
            t.reshape({3, 4});
            t.reshape({2, 2, 3});
            t.reshape({12});
            okay &= equal(t.data(), values);
        }

        if (okay) pass("TC45");
        else fail("TC45", "Data changed during reshape");
    }
    catch (const exception& e) { fail("TC45", e.what()); }

    // TC46: 10,000 create/fill/destroy iterations
    try {
        bool okay = true;

        for (int i = 0; i < 10000; ++i) {
            T t({100, 100});
            t.fill(1.0f);
            okay &= (t.data().front() == 1.0f &&
                     t.data().back() == 1.0f);
        }

        if (okay) pass("TC46");
        else fail("TC46", "Incorrect tensor contents");
    }
    catch (const exception& e) { fail("TC46", e.what()); }

    // TC47: Copy constructor independence
    try {
        T a({2, 2}, {1, 2, 3, 4});
        T b = a;
        b.setElement(99, {0, 0});

        if (equal(a.data(), {1, 2, 3, 4}) &&
            equal(b.data(), {99, 2, 3, 4}))
            pass("TC47");
        else fail("TC47", "Copy is not independent");
    }
    catch (const exception& e) { fail("TC47", e.what()); }

    // TC48: Copy assignment independence
    try {
        T a({2, 2}, {1, 2, 3, 4});
        T b({1}, {9});
        b = a;
        b.setElement(77, {1, 1});

        if (equal(a.data(), {1, 2, 3, 4}) &&
            equal(b.data(), {1, 2, 3, 77}))
            pass("TC48");
        else fail("TC48", "Assignment copy is not independent");
    }
    catch (const exception& e) { fail("TC48", e.what()); }

    // TC49: 100 repeated 500x500 allocation/fill/sum cycles
    try {
        bool okay = true;

        for (int iteration = 1; iteration <= 100; ++iteration) {
            T t({500, 500});
            t.fill(float(iteration));

            double sum = 0;
            for (float x : t.data()) sum += x;

            if (sum != 250000.0 * iteration)
                okay = false;
        }

        if (okay) pass("TC49");
        else fail("TC49", "Incorrect sum");
    }
    catch (const exception& e) { fail("TC49", e.what()); }

    // TC50: Controlled allocation-failure test
    try {
        const size_t huge = static_cast<size_t>(1ULL << 31);
        bool rejected = false;

        try {
            T t(vector<size_t>{huge, huge});
        }
        catch (const length_error&) { rejected = true; }
        catch (const bad_alloc&) { rejected = true; }
        catch (const overflow_error&) { rejected = true; }

        if (rejected) pass("TC50");
        else fail("TC50", "Allocation unexpectedly succeeded");
    }
    catch (const exception& e) { fail("TC50", e.what()); }

    cout << "\n========== SUMMARY ==========\n";
    cout << "Total tests: " << passed + failed + notSupported << '\n';
    cout << "Passed: " << passed << '\n';
    cout << "Failed: " << failed << '\n';
    cout << "Not supported: " << notSupported << '\n';

    return failed == 0 ? 0 : 1;
}
