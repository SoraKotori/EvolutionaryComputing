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

        if (0 == _Iterator % 25)
        {
            _Evolutionary.Perturbing(600);
        }

        auto _Result = _Evolutionary.GetGlobalExtremum();
        _Extremum.push_back(_Result.second);
    }

    _Extremum.shrink_to_fit();
    return _Extremum;
}

template<typename _Type, typename _Function>
void TestGeneticAlgorithm(
    size_t _Count,
    size_t _IteratorCount,
    size_t _Dimension,
    _Type _Min,
    _Type _Max,
    _Function TestFunction)
{
    size_t _ElitismSize = _Count * 0.1;
    vector<double> _CrossoverRate = { 0.2, 0.5, 0.8 };
    vector<double> _MutationRate = { 0.2, 0.5, 0.8 };

    string _FileName = "GeneticAlgorithm"s;
    string _FilePath = _FileName + ".csv";
    ofstream _OutFileStream(_FilePath);
    _OutFileStream << setprecision(32);

    for (auto&& _Crossover : _CrossoverRate)
    {
        for (auto&& _Mutation : _MutationRate)
        {
            GeneticAlgorithm<_Type, less<>> _MyGA(
                _Count,
                _Dimension,
                _ElitismSize,
                _Crossover,
                _Mutation,
                _Min,
                _Max,
                TestFunction,
                random_device()());
            auto&& _MyGAExtremum = Run<_Type>(_IteratorCount, _MyGA);

            copy(begin(_MyGAExtremum), end(_MyGAExtremum), ostream_iterator<double>(_OutFileStream, ","));
            _OutFileStream << endl;
        }
    }
}

template<typename _Type, typename _Function>
void TestDifferentialEvolution(
    size_t _Count,
    size_t _IteratorCount,
    size_t _Dimension,
    _Type _Min,
    _Type _Max,
    _Function TestFunction)
{
    vector<double> _CrossoverProbability = { 0.2, 0.5, 0.8 };
    vector<double> _MutationFactor = { 0.2, 0.5, 0.8 };

    string _FileName = "DifferentialEvolution"s;
    string _FilePath = _FileName + ".csv";
    ofstream _OutFileStream(_FilePath);
    _OutFileStream << setprecision(32);

    for (auto&& _Crossover : _CrossoverProbability)
    {
        for (auto&& _Mutation : _MutationFactor)
        {
            DifferentialEvolution<_Type, less<>> _MyDE(
                _Count,
                _Dimension,
                _Crossover,
                _Mutation,
                _Min,
                _Max,
                TestFunction,
                random_device()());
            auto&& _MyDEExtremum = Run<_Type>(_IteratorCount, _MyDE);

            copy(begin(_MyDEExtremum), end(_MyDEExtremum), ostream_iterator<double>(_OutFileStream, ","));
            _OutFileStream << endl;
        }
    }
}

template<typename _Type, typename _Function>
void TestParticleSwarmOptimization(
    size_t _Count,
    size_t _IteratorCount,
    size_t _Dimension,
    _Type _Min,
    _Type _Max,
    _Function TestFunction)
{
    vector<_Type> _OmegaVector = { 0.2, 0.5, 0.8 };
    vector<_Type> _PhiPVector = { 0.2, 0.5, 0.8 };
    vector<_Type> _PhiGVector = { 0.2, 0.5, 0.8 };

    string _FileName = "ParticleSwarmOptimization"s;
    string _FilePath = _FileName + ".csv";
    ofstream _OutFileStream(_FilePath);
    _OutFileStream << setprecision(32);

    for (auto&& _Omega : _OmegaVector)
    {
        for (auto&& _PhiP : _PhiPVector)
        {
            for (auto&& _PhiG : _PhiGVector)
            {
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
                
                copy(begin(_MyPSOExtremum), end(_MyPSOExtremum), ostream_iterator<double>(_OutFileStream, ","));
                _OutFileStream << endl;
            }
        }
    }
}

template<typename _Type, typename _Function>
void TestALL(
    size_t _Count,
    size_t _IteratorCount,
    size_t _Dimension,
    _Type _Min,
    _Type _Max,
    _Function TestFunction)
{
    size_t _ElitismSize = _Count * 0.1;
    double _CrossoverRate = 0.5;
    double _MutationRate = 0.2;
    GeneticAlgorithm<_Type, less<>> _MyGA(
        _Count,
        _Dimension,
        _ElitismSize,
        _CrossoverRate,
        _MutationRate,
        _Min,
        _Max,
        TestFunction,
        random_device()());
    auto&& _MyGAExtremum = Run<_Type>(_IteratorCount, _MyGA);

    double _CrossoverProbability = 0.5;
    double _MutationFactor = 0.2;
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

    _Type _Omega = 0.5;
    _Type _PhiP = 0.5;
    _Type _PhiG = 0.5;
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

    string _FileName = "ALL"s;
    string _FilePath = _FileName + "_" + to_string(_Count) + "_" + to_string(_IteratorCount) + ".csv";
    ofstream _OutFileStream(_FilePath);
    _OutFileStream << setprecision(32);

    copy(begin(_MyGAExtremum), end(_MyGAExtremum), ostream_iterator<double>(_OutFileStream, ","));
    _OutFileStream << endl;

    copy(begin(_MyDEExtremum), end(_MyDEExtremum), ostream_iterator<double>(_OutFileStream, ","));
    _OutFileStream << endl;

    copy(begin(_MyPSOExtremum), end(_MyPSOExtremum), ostream_iterator<double>(_OutFileStream, ","));
    _OutFileStream << endl;
}

int main(void)
{
    using _Type = double;

    size_t _Count = 2000;
    size_t _IteratorCount = 100;
    size_t _Dimension = 2;
    _Type _Min = -10.0;
    _Type _Max = 10.0;
    auto CrossInTray = [](const vector<_Type>& _Vector)
    {
        auto&& x = _Vector[0];
        auto&& y = _Vector[1];

        return -0.0001 * pow(abs(sin(x) * sin(y) * exp(abs(100 - sqrt(x * x + y * y) / M_PI))) + 1, 0.1);
    };

    auto Matyas = [](const vector<_Type>& _Vector)
    {
        auto&& x = _Vector[0];
        auto&& y = _Vector[1];

        return 0.26 * (x * x + y * y) - 0.48 * x * y;
    };

    //TestGeneticAlgorithm(_Count, _IteratorCount, _Dimension, _Min, _Max, CrossInTray);
    //TestDifferentialEvolution(_Count, _IteratorCount, _Dimension, _Min, _Max, CrossInTray);
    //TestParticleSwarmOptimization(_Count, _IteratorCount, _Dimension, _Min, _Max, CrossInTray);

    TestALL(_Count, _IteratorCount, _Dimension, _Min, _Max, CrossInTray);
    
    return EXIT_SUCCESS;
}