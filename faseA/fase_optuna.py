# Crear scripts basados en Optuna (similares a experiment_2.10.py) para explorar y encontrar 
# configuraciones prometedoras de hiperparámetros para los tres algoritmos (SARSA, Expected SARSA 
# y Q-Learning), utilizando cada uno de los tres presupuestos de discretización (un total de nueve combinaciones).

# Explorar al menos 180 combinaciones de hiperparámetros (20 por algoritmo y presupuesto de discretización), 
# manteniendo el mismo presupuesto de entrenamiento durante los experimentos.

import os
import sys
import random
import optuna

from optuna.samplers import TPESampler

import json

ACTIVIDAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0:0] = [ACTIVIDAD, os.path.dirname(ACTIVIDAD)]

from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.qlearning import QLearning

from config import crear_entorno, ALGORITMOS

MIS_PRESUPUESTOS = sys.argv[1:]  or sys.exit("Uso: python faseA/fase_optuna.py bajo|medio|alto")

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    lr_decay = trial.suggest_float("learning_rate_decay", 0.9 , 1.0,  step=0.01)
    lr_episodes_decay = trial.suggest_categorical("lr_episodes_decay",[100, 1_000, 10_000])
    discount_rate = trial.suggest_float("discount_rate", 0.8, 1.0, step=0.05)
    epsilon= trial.suggest_float("epsilon", 0.0, 0.4, step=0.05)
    
    n_steps = 200_000
    
    if algo_name == "sarsa":
        algo = Sarsa(env)
    elif algo_name == "esarsa": 
        algo = ExpectedSarsa(env)
    elif algo_name == "qlearning": 
        algo = QLearning(env)
        
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    algo.learn(epsilon_greedy_policy,n_steps=n_steps, discount_rate=discount_rate, lr=lr, lrdecay=lr_decay, n_episodes_decay=lr_episodes_decay)
    
    avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100)
    
    return avg_reward

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("./optuna/", exist_ok=True)
    
    for presupuesto in MIS_PRESUPUESTOS:
        for algo_name in ALGORITMOS:
            env = crear_entorno(presupuesto)
            seed = 47
            random.seed(seed)
            env.reset(seed = seed)
            
            storage_file = f"sqlite:///optuna/optuna_{presupuesto}.db"
            study_name = f"{algo_name}_{presupuesto}"
            full_study_dir_path = f"optuna/{study_name}"
            tpe_sampler = TPESampler(seed=seed)
            study = optuna.create_study(sampler=tpe_sampler, direction='maximize', study_name=study_name, storage=storage_file, load_if_exists=True)
            
            n_trials = 20
            
            print(f"[{study_name}] Searching for the best hyperparameters in {n_trials} trials...")
            
            study.optimize(objective, n_trials=n_trials)
            
            env.close()
            
            best_trial = study.best_trial
            best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
            
            os.makedirs(full_study_dir_path, exist_ok=True)
            best_trial_file = open(f"{full_study_dir_path}/best_trial.json", "w")
            best_trial_file.write(best_trial_params)
            best_trial_file.close()
            
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(f"{full_study_dir_path}/optimization_history.html")
            fig = optuna.visualization.plot_contour(study)
            fig.write_html(f"{full_study_dir_path}/contour.html")
            fig = optuna.visualization.plot_slice(study)
            fig.write_html(f"{full_study_dir_path}/slice.html")
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(f"{full_study_dir_path}/param_importances.html")