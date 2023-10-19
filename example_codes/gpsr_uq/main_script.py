import numpy as np

from bingo.evolutionary_algorithms.generalized_crowding import \
                                                GeneralizedCrowdingEA
from bingo.selection.bayes_crowding import BayesCrowding
from bingo.selection.deterministic_crowding import DeterministicCrowding
from bingo.evolutionary_optimizers.parallel_archipelago import \
                                            ParallelArchipelago, \
                                            load_parallel_archipelago_from_file
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt \
                               import ContinuousLocalOptimization

from bingo.stats.pareto_front import ParetoFront

from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.symbolic_regression.bayes_fitness.bayes_fitness_function \
                               import BayesFitnessFunction
from bingo.symbolic_regression.agraph.agraph import AGraph

POP_SIZE = 400
STACK_SIZE = 40
MAX_GEN = 10000
FIT_THRESH = -np.inf
CHECK_FREQ = 50
MIN_GEN = 500
VARIANCE_CAPS = [None]*6 + [70]*8

def make_training_data():

    model = AGraph(equation="3*sin(X_0) + 2*X_0")
    X = np.linspace(0, np.pi, 25).reshape((-1,1))
    y = model.evaluate_equation_at(X) + \
            np.random.normal(0, 0.5, size=X.shape)
    training_data = ExplicitTrainingData(x=X, y=y)
    
    return training_data


def execute_generational_steps():

    training_data = make_training_data()

    component_generator = ComponentGenerator(training_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=True
                                       )

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')

    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)

    smc_hyperparams = {'num_particles':10,
                       'mcmc_steps':10,
                       'ess_threshold':0.75}

    bff = BayesFitnessFunction(local_opt_fitness,
                               smc_hyperparams)

    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)

    evaluator = Evaluation(bff, multiprocess=8)

    selection_phase=BayesCrowding()
    ea = GeneralizedCrowdingEA(evaluator, crossover,
                      mutation, 0.4, 0.4, selection_phase)


    island = Island(ea, agraph_generator, POP_SIZE)
    opt_result = island.evolve_until_convergence(max_generations=MAX_GEN,
                                                  fitness_threshold=FIT_THRESH,
                                        convergence_check_frequency=CHECK_FREQ,
                                              checkpoint_base_name='checkpoint')

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and \
                            ag_1.get_complexity() == ag_2.get_complexity()

if __name__ == '__main__':
    execute_generational_steps()

