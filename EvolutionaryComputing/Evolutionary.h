#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace evolutionary
{
    template<typename _Type>
    using _VectorType = std::vector<_Type>;

    template<typename _Type, typename _Comparison, typename _EngineType = std::default_random_engine>
    class Evolutionary
    {
    public:
        using _SizeType = typename _VectorType<_Type>::size_type;
        using _ValueType = typename _VectorType<_Type>::value_type;
        using _FunctionType = _ValueType(*)(const _VectorType<_ValueType>&);
        using _ComparisonType = _Comparison;

        virtual void Perturbing(_SizeType _Size) = 0;
        virtual void Update(void) = 0;
        virtual std::pair<const _VectorType<_ValueType>&, _ValueType> GetGlobalExtremum(void) = 0;

    protected:
        template<typename... _Args>
        Evolutionary(_ValueType _Min, _ValueType _Max, _FunctionType Function, _Args&&... __args) :
            _DomainUniform(_Min, _Max),
            FitnessFunction(Function),
            _Engine(std::forward<_Args>(__args)...)
        {
        }

        // Do not use
        decltype(auto) Fitness(const _VectorType<_VectorType<_ValueType>>& _Domain)
        {
            _VectorType<_ValueType> _Fitness(_Domain.size());
            std::transform(std::begin(_Domain), std::end(_Domain), std::begin(_Fitness), FitnessFunction);

            return _Fitness;
        }

        // There may be a better way
        template<typename _Compare>
        void Sort(
            _VectorType<_ValueType>& _Fitness,
            _VectorType<_VectorType<_ValueType>>& _Population,
            _Compare _Comp)
        {
            _VectorType<std::pair<_ValueType, _VectorType<_ValueType>>> _PairVector;
            _PairVector.reserve(_Fitness.size());

            std::transform(std::begin(_Fitness), std::end(_Fitness), std::begin(_Population), std::back_inserter(_PairVector),
                std::make_pair<std::reference_wrapper<_ValueType>, std::reference_wrapper<_VectorType<_ValueType>>>);

            std::sort(std::begin(_PairVector), std::end(_PairVector),
                [&](auto&& _Left, auto&& _Right) { return _Comp(_Left.first, _Right.first); });

            std::transform(std::begin(_PairVector), std::end(_PairVector), std::begin(_Fitness),
                [](auto&& _Pair) { return _Pair.first; });

            std::transform(std::begin(_PairVector), std::end(_PairVector), std::begin(_Population),
                [](auto&& _Pair) { return _Pair.second; });
        }

        template<typename _ForwardIterator, typename _DistributionType>
        void Reset(_ForwardIterator&& _First, _ForwardIterator&& _Last, _DistributionType&& _Distribution)
        {
            std::for_each(_First, _Last, [&](_VectorType<_ValueType>& _Vector)
            {
                std::generate(std::begin(_Vector), std::end(_Vector), std::bind(_Distribution, std::ref(_Engine)));
            });
        }

        template<typename _InputIterator, typename _Compare>
        decltype(auto) Search(_InputIterator&& _First, _InputIterator&& _Last, _Compare _Comp)
        {
            auto _BestVector = std::ref(*_First++);
            auto _BestFitness = FitnessFunction(_BestVector);
            std::for_each(_First, _Last, [&, _Comp](auto&& _Vector)
            {
                auto&& _Fitness = this->FitnessFunction(_Vector);
                if (_Comp(_Fitness, _BestFitness))
                {
                    _BestVector = _Vector;
                    _BestFitness = _Fitness;
                }
            });

            return std::make_pair(_BestVector, _BestFitness);
        }

        std::uniform_real_distribution<_ValueType> _DomainUniform;
        _FunctionType FitnessFunction;
        _EngineType _Engine;
    };

    template<typename... _BaseArgs>
    class DifferentialEvolution : public Evolutionary<_BaseArgs...>
    {
    protected:
        using _BaseType = Evolutionary<_BaseArgs...>;
        using _BaseType::_DomainUniform;
        using _BaseType::FitnessFunction;
        using _BaseType::_Engine;
        using _BaseType::Sort;
        using _BaseType::Reset;
        using _BaseType::Search;

    public:
        using typename _BaseType::_SizeType;
        using typename _BaseType::_ValueType;
        using typename _BaseType::_ComparisonType;

        template<typename... _Args>
        DifferentialEvolution(
            _SizeType _Count,
            _SizeType _Dimension,
            double _CrossoverProbability,
            _ValueType _DifferentialWeight,
            _ValueType _Min,
            _ValueType _Max,
            _Args&&... __args
        ) :
            _BaseType(_Min, _Max, std::forward<_Args>(__args)...),
            _Population(_Count, _VectorType<_ValueType>(_Dimension)),
            _Candidate(_Dimension),
            _PopulationShuffle(_Count),
            _IndexDistribution(_SizeType(0), _Count - 1),
            _DimensionDistribution(_SizeType(0), _Dimension - 1),
            _CrossoverDistribution(_CrossoverProbability),
            _MutationFactor(_DifferentialWeight)
        {
            Reset(std::begin(_Population), std::end(_Population), _DomainUniform);
            std::iota(std::begin(_PopulationShuffle), std::end(_PopulationShuffle), _SizeType(0));
        }

        void Perturbing(_SizeType _Size) override
        {
            using namespace std::placeholders;

            _VectorType<_ValueType> _Fitness(_Population.size());
            std::transform(std::begin(_Population), std::end(_Population), std::begin(_Fitness), FitnessFunction);

            Sort(_Fitness, _Population, _ComparisonType());
            std::reverse(std::begin(_Population), std::end(_Population));

            Reset(std::begin(_Population), std::next(std::begin(_Population), _Size), _DomainUniform);
        }

        void Update() override
        {
            std::shuffle(std::begin(_PopulationShuffle), std::end(_PopulationShuffle), _Engine);

            auto&& _Count = _Population.size();
            for (decltype(_Count) _Index = 0; _Index < _Count; ++_Index)
            {
                auto&& _PopulationIndex = _PopulationShuffle[_Index];
                auto&& _Random = GetRandomArray(_PopulationIndex);

                auto&& _Original = _Population[_PopulationIndex];
                auto&& _A = _Population[_Random[0]];
                auto&& _B = _Population[_Random[1]];
                auto&& _C = _Population[_Random[2]];

                auto&& _DimensionSelect = _DimensionDistribution(_Engine);
                auto&& _DimensionCount = _Candidate.size();
                for (decltype(_DimensionCount) _Dimension(0); _Dimension < _DimensionCount; ++_Dimension)
                {
                    auto&& _CrossoverBool = _CrossoverDistribution(_Engine);
                    auto&& _DimensionBool = _DimensionSelect == _Dimension;
                    auto&& _TotelBool = _CrossoverBool || _DimensionBool;

                    _Candidate[_Dimension] =
                        _TotelBool ? _A[_Dimension] + _MutationFactor * (_B[_Dimension] - _C[_Dimension]) : _Original[_Dimension];
                }

                auto&& _OriginalFitness = FitnessFunction(_Original);
                auto&& _CandidateFitness = FitnessFunction(_Candidate);
                if (_ComparisonType()(_CandidateFitness, _OriginalFitness))
                {
                    std::swap(_Original, _Candidate);
                }
            }
        }

        std::pair<const _VectorType<_ValueType>&, _ValueType> GetGlobalExtremum(void) override
        {
            return Search(std::begin(_Population), std::end(_Population), _ComparisonType());
        }

    private:
        _VectorType<_VectorType<_ValueType>> _Population;
        _VectorType<_ValueType> _Candidate;
        _VectorType<_SizeType> _PopulationShuffle;

        std::uniform_int_distribution<_SizeType> _IndexDistribution;
        std::uniform_int_distribution<_SizeType> _DimensionDistribution;
        std::bernoulli_distribution _CrossoverDistribution;
        _ValueType _MutationFactor;

        decltype(auto) GetRandomArray(_SizeType _PopulationIndex)
        {
            using namespace std::placeholders;
            std::array<_SizeType, 3> _Random;

            auto&& _First = std::begin(_Random);
            auto&& _Last = std::end(_Random);
            auto _Middle = _First;

            while (_Middle != _Last)
            {
                auto&& _Index = _IndexDistribution(_Engine);
                if (_PopulationIndex != _Index)
                {
                    if (std::all_of(_First, _Middle, std::bind(std::not_equal_to<_SizeType>(), _1, _Index)))
                    {
                        *_Middle++ = _Index;
                    }
                }
            }

            return _Random;
        }
    };

    template<typename... _BaseArgs>
    class ParticleSwarmOptimization : public Evolutionary<_BaseArgs...>
    {
    protected:
        using _BaseType = Evolutionary<_BaseArgs...>;
        using _BaseType::_DomainUniform;
        using _BaseType::FitnessFunction;
        using _BaseType::_Engine;
        using _BaseType::Sort;
        using _BaseType::Reset;
        using _BaseType::Search;

    public:
        using typename _BaseType::_SizeType;
        using typename _BaseType::_ValueType;
        using typename _BaseType::_ComparisonType;

        template<typename... _Args>
        ParticleSwarmOptimization(
            _SizeType _Count,
            _SizeType _Dimension,
            _ValueType __omega,
            _ValueType __phiP,
            _ValueType __phiG,
            _ValueType _Min,
            _ValueType _Max,
            _Args&&... __args
        ) :
            _BaseType(_Min, _Max, std::forward<_Args>(__args)...),
            _Position(_Count, _VectorType<_ValueType>(_Dimension)),
            _Velocity(_Count, _VectorType<_ValueType>(_Dimension)),
            _ParticleBest(_Count, _VectorType<_ValueType>(_Dimension)),
            _SwarmBest(_ParticleBest[0]),
            _Omega(__omega),
            _PhiP(__phiP),
            _PhiG(__phiG),
            _VelocityDistribution(-std::abs(_Max - _Min), std::abs(_Max - _Min))
        {
            // Initialize the particle's position with a uniformly distributed random vector: xi ~ U(blo, bup)
            Reset(std::begin(_Position), std::end(_Position), _DomainUniform);

            // Initialize the particle's best known position to its initial position: pi ¡ö xi
            _ParticleBest = _Position;

            // if f(pi) < f(g) then update the swarm's best known  position: g ¡ö pi
            auto&& Result = Search(std::begin(_Position), std::end(_Position), _ComparisonType());
            _SwarmBest = Result.first;
            _SwarmBestFitness = Result.second;

            // Initialize the particle's velocity: vi ~ U(-|bup-blo|, |bup-blo|)
            Reset(std::begin(_Velocity), std::end(_Velocity), _VelocityDistribution);
        }

        void Perturbing(_SizeType _Size) override
        {
            using namespace std::placeholders;

            _VectorType<_ValueType> _Fitness(_Position.size());
            std::transform(std::begin(_Position), std::end(_Position), std::begin(_Fitness), FitnessFunction);

            Sort(_Fitness, _Position, _ComparisonType());
            std::reverse(std::begin(_Position), std::end(_Position));

            Reset(std::begin(_Position), std::next(std::begin(_Position), _Size), _DomainUniform);

            // Unfinished
        }

        void Update() override
        {
            auto&& _Count = _Position.size();
            auto&& _Dimension = _Position[0].size();

            // for each particle i = 1, ..., S do
            for (decltype(_Count) _ParticleIndex{ 0 }; _ParticleIndex < _Count; ++_ParticleIndex)
            {
                auto&& _PositionVector = _Position[_ParticleIndex];
                auto&& _VelocityVector = _Velocity[_ParticleIndex];
                auto&& _ParticleBestVector = _ParticleBest[_ParticleIndex];

                // for each dimension d = 1, ..., n do
                for (decltype(_Dimension) _DimensionIndex{ 0 }; _DimensionIndex < _Dimension; ++_DimensionIndex)
                {
                    auto&& _PositionValue = _PositionVector[_DimensionIndex];
                    auto&& _VelocityValue = _VelocityVector[_DimensionIndex];
                    auto&& _ParticleBestValue = _ParticleBestVector[_DimensionIndex];
                    auto&& _SwarmBestValue = _SwarmBest.get()[_DimensionIndex];

                    // Pick random numbers: rp, rg ~ U(0,1)
                    auto&& _RandomR = std::generate_canonical<_ValueType, std::numeric_limits<_ValueType>::digits>(_Engine);
                    auto&& _RandomG = std::generate_canonical<_ValueType, std::numeric_limits<_ValueType>::digits>(_Engine);

                    // Update the particle's velocity: vi,d ¡ö £s vi,d + £pp rp (pi,d-xi,d) + £pg rg (gd-xi,d)
                    _VelocityValue =
                        _Omega * _VelocityValue +
                        _PhiP * _RandomR * (_ParticleBestValue - _PositionValue) +
                        _PhiG * _RandomG * (_SwarmBestValue - _PositionValue);

                    // Update the particle's position: xi ¡ö xi + vi
                    _PositionValue += _VelocityValue;
                }

                auto&& _PositionFitness = FitnessFunction(_PositionVector);
                auto&& _ParticleBestFitness = FitnessFunction(_ParticleBestVector);

                // if f(xi) < f(pi) then
                if (_ComparisonType()(_PositionFitness, _ParticleBestFitness))
                {
                    // Update the particle's best known position: pi ¡ö xi
                    _ParticleBestVector = _PositionVector;

                    // if f(pi) < f(g) then
                    if (_ComparisonType()(_PositionFitness, _SwarmBestFitness))
                    {
                        // Update the swarm's best known position: g ¡ö pi
                        _SwarmBestFitness = _PositionFitness;
                        _SwarmBest = _PositionVector;
                    }
                }
            }
        }

        std::pair<const _VectorType<_ValueType>&, _ValueType> GetGlobalExtremum(void) override
        {
            return std::make_pair(_SwarmBest, _SwarmBestFitness);
        }

    private:
        _VectorType<_VectorType<_ValueType>> _Position;
        _VectorType<_VectorType<_ValueType>> _Velocity;

        _VectorType<_VectorType<_ValueType>> _ParticleBest;
        std::reference_wrapper<_VectorType<_ValueType>> _SwarmBest;
        _ValueType _SwarmBestFitness;

        _ValueType _Omega;
        _ValueType _PhiP;
        _ValueType _PhiG;

        std::uniform_real_distribution<_ValueType> _VelocityDistribution;
    };

    template<typename... _BaseArgs>
    class GeneticAlgorithm : public Evolutionary<_BaseArgs...>
    {
    protected:
        using _BaseType = Evolutionary<_BaseArgs...>;
        using _BaseType::_DomainUniform;
        using _BaseType::FitnessFunction;
        using _BaseType::_Engine;
        using _BaseType::Sort;
        using _BaseType::Reset;
        using _BaseType::Search;

    public:
        using typename _BaseType::_SizeType;
        using typename _BaseType::_ValueType;
        using typename _BaseType::_ComparisonType;

        template<typename... _Args>
        GeneticAlgorithm(
            _SizeType _PopulationSize,
            _SizeType _Dimension,
            _SizeType _ElitismSize,
            double _CrossoverRate,
            double _MutationRate,
            _ValueType _Min,
            _ValueType _Max,
            _Args&&... __args
        ) :
            _BaseType(_Min, _Max, std::forward<_Args>(__args)...),
            _Parent(_PopulationSize, _VectorType<_ValueType>(_Dimension)),
            _Child(_PopulationSize, _VectorType<_ValueType>(_Dimension)),
            _ParentFitness(_PopulationSize),
            _ChildFitness(_PopulationSize),
            _Elitism(_ElitismSize),
            _Shift(_Min),
            _Best(_Parent[0]),
            _CrossoverDistribution(_CrossoverRate),
            _MutationDistribution(_MutationRate),
            _DimensionDistribution(_SizeType(0), _Dimension - 1)
        {
            Reset(std::begin(_Child), std::end(_Child), _DomainUniform);
            EvaluateFitness(_SizeType(0));
        }

        void Perturbing(_SizeType _Size) override
        {
            using namespace std::placeholders;

            Sort(_ParentFitness, _Parent, _ComparisonType());
            std::reverse(std::begin(_Parent), std::end(_Parent));
            std::reverse(std::begin(_ParentFitness), std::end(_ParentFitness));

            Reset(std::begin(_Parent), std::next(std::begin(_Parent), _Size), _DomainUniform);
            std::transform(std::begin(_Parent), std::next(std::begin(_Parent), _Size), std::begin(_ParentFitness), FitnessFunction);
        }

        void Update() override
        {
            RouletteWheelSelection();
            SinglePointCrossover();
            Mutation();
            EvaluateFitness(_Elitism);
        }

        std::pair<const _VectorType<_ValueType>&, _ValueType> GetGlobalExtremum(void) override
        {
            return std::make_pair(_Best, _BestFitness);
        }

    private:
        _VectorType<_VectorType<_ValueType>> _Parent;
        _VectorType<_VectorType<_ValueType>> _Child;
        _VectorType<_ValueType> _ParentFitness;
        _VectorType<_ValueType> _ChildFitness;

        _SizeType _Elitism;
        _ValueType _Shift;

        _ValueType _BestFitness;
        std::reference_wrapper<_VectorType<_ValueType>> _Best;

        std::bernoulli_distribution _CrossoverDistribution;
        std::bernoulli_distribution _MutationDistribution;
        std::uniform_int_distribution<_SizeType> _DimensionDistribution;

        void EvaluateFitness(_SizeType _SwapSize)
        {
            using namespace std::placeholders;
            std::transform(std::begin(_Child), std::end(_Child), std::begin(_ChildFitness), FitnessFunction);
            std::transform(std::begin(_Parent), std::end(_Parent), std::begin(_ParentFitness), FitnessFunction);

            Sort(_ChildFitness, _Child, _ComparisonType());
            Sort(_ParentFitness, _Parent, _ComparisonType());
            std::reverse(std::begin(_Parent), std::end(_Parent));
            std::reverse(std::begin(_ParentFitness), std::end(_ParentFitness));
            std::swap_ranges(std::begin(_Parent), std::prev(std::end(_Parent), _SwapSize), std::begin(_Child));
            std::swap_ranges(std::begin(_ParentFitness), std::prev(std::end(_ParentFitness), _SwapSize), std::begin(_ChildFitness));

            auto&& _BestIterator = std::min_element(std::begin(_ParentFitness), std::end(_ParentFitness), _ComparisonType());
            _BestFitness = *_BestIterator;
            _Best = _Parent[std::distance(std::begin(_ParentFitness), _BestIterator)];
        }

        void RouletteWheelSelection()
        {
            if (0 > _BestFitness)
            {
                std::transform(std::begin(_ParentFitness), std::end(_ParentFitness), std::begin(_ParentFitness), std::negate<>());
            }
            else
            {
                using namespace std::placeholders;
                std::transform(std::begin(_ParentFitness), std::end(_ParentFitness), std::begin(_ParentFitness), std::bind(std::divides<>(), 1, _1));
            }

            // It can't be less than 0 
            std::discrete_distribution<_SizeType> _ParentDistribution(std::begin(_ParentFitness), std::end(_ParentFitness));
            for (auto&& _Chromosome : _Child)
            {
                auto&& _ParentIndex = _ParentDistribution(_Engine);
                _Chromosome = _Parent[_ParentIndex];
            }
        }

        void SinglePointCrossover()
        {
            auto&& _First = std::begin(_Child);
            auto&& _Last = std::end(_Child);

            for (; _First != _Last; _First = std::next(_First, 2))
            {
                auto&& _Next = std::next(_First);
                if (_Next == _Last)
                {
                    break;
                }

                auto&& _Crossover = _CrossoverDistribution(_Engine);
                if (_Crossover)
                {
                    auto&& _DimensionIndex = _DimensionDistribution(_Engine);
                    std::swap_ranges(std::begin(*_First), std::next(std::begin(*_First), _DimensionIndex), std::begin(*_Next));

                    auto&& _FirstValue = (*_First)[_DimensionIndex];
                    auto&& _NextValue = (*_Next)[_DimensionIndex];

                    auto&& _Mask = _DomainUniform(_Engine) - _Shift;
                    auto&& _FirstRemainder = fmod(_FirstValue - _Shift, _Mask);
                    auto&& _NextRemainder = fmod(_NextValue - _Shift, _Mask);

                    _FirstValue = _FirstValue - _FirstRemainder + _NextRemainder;
                    _NextValue = _NextValue - _NextRemainder + _FirstRemainder;
                }
            }
        }

        void Mutation()
        {
            for (auto&& _Vector : _Child)
            {
                for (auto&& _Value : _Vector)
                {
                    auto&& _Mutation = _MutationDistribution(_Engine);
                    if (_Mutation)
                    {
                        _Value += _DomainUniform(_Engine);
                    }
                }
            }
        }
    };
}