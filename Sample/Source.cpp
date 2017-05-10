#define _USE_MATH_DEFINES
#include "Evolutionary.h"
#include <array>
#include <fstream>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <string>

using namespace std;
using namespace evolutionary;

template<typename _Type, typename... _BaseArgs>
decltype(auto) Run(size_t _IteratorCount, Evolutionary<_BaseArgs...>& _Evolutionary)
{
    _VectorType<typename Evolutionary<_BaseArgs...>::_ValueType> _Extremum;
    _Extremum.reserve(_IteratorCount);

    size_t _Iterator = 0;
    for (; _Iterator < _IteratorCount; ++_Iterator)
    {
        _Evolutionary.Update();

        if (0 == _Iterator % 100)
        {
            cout << _Iterator << endl;
        }

        if (0 == _Iterator % 100)
        {
            _Evolutionary.Perturbing(30);
        }

        auto _Result = _Evolutionary.GetGlobalExtremum();
        _Extremum.push_back(_Result.second);
    }

    _Extremum.shrink_to_fit();
    return _Extremum;
}

int main(void)
{
    using _Type = double;

    size_t _Count = 2000;
    size_t _IteratorCount = 10000;
    size_t _Dimension = 2;
    _Type _Min = -10.0;
    _Type _Max = 10.0;
    auto TestFunction = [](const vector<_Type>& _Vector)
    {
        auto&& x = _Vector[0];
        auto&& y = _Vector[1];

        return -0.0001 * pow(abs(sin(x) * sin(y) * exp(abs(100 - sqrt(x * x + y * y) / M_PI))) + 1, 0.1);
    };

    // DifferentialEvolution
    double _CrossoverProbability = 0.5;
    _Type _MutationFactor = 1.0;
    DifferentialEvolution<_Type, less<>> _MyDE(
        _Count,
        _Dimension,
        _CrossoverProbability,
        _MutationFactor,
        _Min,
        _Max,
        TestFunction,
        random_device()());
    auto&& _MyDEExtremum = Run<_Type>(_IteratorCount, _MyDE);


    // ParticleSwarmOptimization
    _Type _Omega = 0.1;
    _Type _PhiP = 0.1;
    _Type _PhiG = 0.1;
    ParticleSwarmOptimization<_Type, less<>> _MyPSO(
        _Count,
        _Dimension,
        _Omega,
        _PhiP,
        _PhiG,
        _Min,
        _Max,
        TestFunction,
        random_device()());
    auto&& _MyPSOExtremum = Run<_Type>(_IteratorCount, _MyPSO);


    // GeneticAlgorithm
    size_t _ElitismSize = 10;
    double _CrossoverRate = 1.0;
    double _MutationRate = 0.1;
    GeneticAlgorithm<_Type, less<>> _MyGA(
        _Count,
        _Dimension,
        _ElitismSize,
        _CrossoverRate,
        _MutationRate,
        _Min,
        _Max,
        TestFunction);
    auto&& _MyGAExtremum = Run<_Type>(_IteratorCount, _MyGA);

    // End
    string _FileName = "Test"s;
    string _FilePath = _FileName + "_" + to_string(_Count) + "_" + to_string(_IteratorCount) + ".csv";

    ofstream _OutFileStream(_FilePath);
    _OutFileStream << setprecision(32);

    copy(begin(_MyDEExtremum), end(_MyDEExtremum), ostream_iterator<double>(_OutFileStream, ","));
    _OutFileStream << endl;

    copy(begin(_MyPSOExtremum), end(_MyPSOExtremum), ostream_iterator<double>(_OutFileStream, ","));
    _OutFileStream << endl;

    copy(begin(_MyGAExtremum), end(_MyGAExtremum), ostream_iterator<double>(_OutFileStream, ","));
    _OutFileStream << endl;

    return EXIT_SUCCESS;
}